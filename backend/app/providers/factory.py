
from app.config import settings
from app.providers.base import AIProvider
from app.providers.gemini import GeminiProvider


def get_ai_provider() -> AIProvider:
    provider = settings.AI_PROVIDER.lower()

    if provider == "gemini":
        return GeminiProvider(
            api_key=settings.GEMINI_API_KEY,
            model=settings.GEMINI_MODEL,
        )

    raise ValueError(
        f"Unsupported AI provider: {provider}"
    )
