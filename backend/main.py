from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.db.session import Base, engine
from app.routes.conversations import router as conversations_router
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.ai_service import AIService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure database tables exist at application startup
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="AI Chat App", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount new conversation history router
app.include_router(conversations_router)

ai_service = AIService()


@app.get("/")
def home():
    return {"message": "AI Chat App Backend is running!"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Legacy chat endpoint preserved for backwards compatibility."""
    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty",
        )

    try:
        reply = await ai_service.chat(request.message)
        return ChatResponse(reply=reply)

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="AI provider request failed",
        )
