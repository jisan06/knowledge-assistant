from abc import ABC, abstractmethod
from typing import AsyncGenerator


class AIProvider(ABC):

    @abstractmethod
    async def generate(self, message: str) -> str:
        """Generate a complete response from an AI provider."""
        pass

    @abstractmethod
    async def generate_stream(self, message: str) -> AsyncGenerator[str, None]:
        """Asynchronously stream chunks of response text from an AI provider."""
        pass
