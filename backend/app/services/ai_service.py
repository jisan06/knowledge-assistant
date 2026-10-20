from typing import AsyncGenerator
from app.providers.factory import get_ai_provider


class AIService:

    def __init__(self):
        self.provider = get_ai_provider()

    async def chat(self, message: str) -> str:
        return await self.provider.generate(message)

    async def chat_stream(self, message: str) -> AsyncGenerator[str, None]:
        async for chunk in self.provider.generate_stream(message):
            yield chunk
