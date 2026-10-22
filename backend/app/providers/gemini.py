import asyncio
import logging
import re
from typing import AsyncGenerator, Optional
from google import genai

from app.providers.base import AIProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain Exceptions for AI / Gemini Providers
# ---------------------------------------------------------------------------

class GeminiAPIError(Exception):
    """Base user-friendly exception for AI service errors."""

    def __init__(
        self,
        message: str = "The AI service encountered an error processing your request.",
        status_code: int = 502,
        is_retryable: bool = False,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.is_retryable = is_retryable


class GeminiRateLimitError(GeminiAPIError):
    """Raised when the AI model rate limit (429) or quota is exhausted."""

    def __init__(
        self,
        message: str = "The AI service is currently experiencing high demand or rate limits. Please try again shortly.",
    ):
        super().__init__(message=message, status_code=429, is_retryable=True)


class GeminiServiceUnavailableError(GeminiAPIError):
    """Raised when the AI service is temporarily unavailable (503 / 502 / timeouts)."""

    def __init__(
        self,
        message: str = "The AI service is temporarily unavailable. Please try again in a few moments.",
    ):
        super().__init__(message=message, status_code=503, is_retryable=True)


class GeminiAuthError(GeminiAPIError):
    """Raised when the API key or authentication credentials fail (401 / 403)."""

    def __init__(
        self,
        message: str = "AI service authentication failed. Please check server configuration.",
    ):
        super().__init__(message=message, status_code=401, is_retryable=False)


# ---------------------------------------------------------------------------
# Retry and Sanitization Helpers
# ---------------------------------------------------------------------------

def extract_retry_after(exc: Exception) -> Optional[float]:
    """
    Extracts retry-after delay in seconds from exception headers or message text.
    Supports integer and decimal second specifications.
    """
    # 1. Inspect direct or response headers
    headers = getattr(exc, "headers", None)
    if not headers and hasattr(exc, "response"):
        headers = getattr(exc.response, "headers", None)
    if headers and hasattr(headers, "items"):
        for key, value in headers.items():
            if str(key).lower() == "retry-after":
                try:
                    val = float(value)
                    if val >= 0:
                        return val
                except (ValueError, TypeError):
                    pass

    # 2. Inspect text pattern in exception message
    msg = str(exc)
    patterns = [
        r"retry\s*[-_]?after\s*[:=]?\s*([0-9.]+)",
        r"retry\s+in\s+([0-9.]+)\s*s",
        r"try\s+again\s+in\s+([0-9.]+)\s*s",
    ]
    for pat in patterns:
        match = re.search(pat, msg, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1))
                if val >= 0:
                    return val
            except (ValueError, TypeError):
                pass

    return None


def is_retryable_error(exc: Exception) -> bool:
    """
    Determines whether an exception is transient and eligible for retry.
    Permanent errors (400, 401, 403, 404) are NOT retried.
    Transient errors (429, 500, 502, 503, 504, connection timeouts) ARE retried.
    """
    if isinstance(exc, (asyncio.TimeoutError, ConnectionError, TimeoutError)):
        return True

    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code is not None:
        try:
            c = int(code)
            if c in (400, 401, 403, 404):
                return False
            if c in (429, 500, 502, 503, 504):
                return True
        except (ValueError, TypeError):
            pass

    err_str = str(exc).lower()

    # Explicit permanent indicators
    permanent_indicators = [
        "400", "invalid_argument", "bad request",
        "401", "unauthenticated", "invalid api key", "api_key_invalid",
        "403", "permission_denied",
        "404", "not_found", "model not found",
    ]
    if any(p in err_str for p in permanent_indicators):
        return False

    # Transient indicators
    transient_indicators = [
        "429", "resource_exhausted", "quota", "rate limit", "too many requests",
        "503", "unavailable", "service unavailable", "overloaded",
        "502", "bad gateway",
        "504", "gateway timeout",
        "timeout", "timed out", "connection reset", "connection refused",
    ]
    return any(t in err_str for t in transient_indicators)


def sanitize_log_message(msg: str) -> str:
    """Removes API keys and sensitive tokens from log messages."""
    sanitized = re.sub(r"AIza[0-9A-Za-z-_]{35}", "[REDACTED_API_KEY]", msg)
    sanitized = re.sub(
        r"(api[-_]?key|key|token|secret)[:=]\s*['\"]?[a-zA-Z0-9_\-\.]{8,}['\"]?",
        r"\1=[REDACTED]",
        sanitized,
        flags=re.IGNORECASE,
    )
    return sanitized


# ---------------------------------------------------------------------------
# Gemini Provider Implementation
# ---------------------------------------------------------------------------

class GeminiProvider(AIProvider):
    """
    Robust Gemini AI Provider with bounded retries, exponential backoff,
    Retry-After parsing, client cancellation support, and strict error sanitization.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 15.0,
    ):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay

    def _raise_sanitized_error(self, exc: Exception) -> None:
        """
        Translates raw API errors into clean, sanitized domain exceptions
        without exposing internal URLs, stack traces, or credentials.
        """
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        err_str = str(exc).lower()

        if code == 429 or any(t in err_str for t in ["429", "resource_exhausted", "quota", "rate limit", "too many requests"]):
            raise GeminiRateLimitError()
        elif code in (502, 503, 504) or any(t in err_str for t in ["503", "unavailable", "service unavailable", "overloaded", "502", "504", "timeout"]):
            raise GeminiServiceUnavailableError()
        elif code in (401, 403) or any(t in err_str for t in ["401", "403", "unauthenticated", "permission_denied", "api key"]):
            raise GeminiAuthError()
        else:
            raise GeminiAPIError(
                message="The AI service encountered an error processing your request. Please try again.",
                status_code=502,
                is_retryable=False,
            )

    async def generate(self, message: str) -> str:
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=message,
                )
                return response.text or ""
            except Exception as exc:
                last_error = exc
                if not is_retryable_error(exc) or attempt >= self.max_retries:
                    break

                retry_after = extract_retry_after(exc)
                if retry_after is not None and 0 < retry_after <= self.max_delay:
                    delay = retry_after
                else:
                    delay = min(self.initial_delay * (self.backoff_factor ** attempt), self.max_delay)

                safe_msg = sanitize_log_message(str(exc))
                logger.warning(
                    f"Gemini API transient failure (attempt {attempt + 1}/{self.max_retries + 1}): {safe_msg}. "
                    f"Retrying in {delay:.2f}s..."
                )
                await asyncio.sleep(delay)

        self._raise_sanitized_error(last_error)

    async def generate_stream(self, message: str) -> AsyncGenerator[str, None]:
        last_error = None
        stream = None

        for attempt in range(self.max_retries + 1):
            try:
                stream = await self.client.aio.models.generate_content_stream(
                    model=self.model,
                    contents=message,
                )
                break
            except Exception as exc:
                last_error = exc
                if not is_retryable_error(exc) or attempt >= self.max_retries:
                    break

                retry_after = extract_retry_after(exc)
                if retry_after is not None and 0 < retry_after <= self.max_delay:
                    delay = retry_after
                else:
                    delay = min(self.initial_delay * (self.backoff_factor ** attempt), self.max_delay)

                safe_msg = sanitize_log_message(str(exc))
                logger.warning(
                    f"Gemini API stream initiation failed (attempt {attempt + 1}/{self.max_retries + 1}): {safe_msg}. "
                    f"Retrying in {delay:.2f}s..."
                )
                await asyncio.sleep(delay)

        if stream is None:
            self._raise_sanitized_error(last_error)

        try:
            async for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except asyncio.CancelledError:
            logger.info("Chat stream was cancelled by client.")
            raise
        except Exception as exc:
            safe_err = sanitize_log_message(str(exc))
            logger.error(f"Error during stream consumption: {safe_err}")
            self._raise_sanitized_error(exc)
