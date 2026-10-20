import json
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.session import get_db, SessionLocal
from app.schemas.chat import (
    ConversationCreate,
    ConversationUpdate,
    ConversationResponse,
    MessageCreate,
    MessageResponse,
    SendMessageResponse,
)
from app.services.chat_history_service import ChatHistoryService
from app.services.ai_service import AIService

router = APIRouter(prefix="/api/conversations", tags=["conversations"])
ai_service = AIService()


@router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
def create_conversation(
    payload: ConversationCreate = ConversationCreate(),
    db: Session = Depends(get_db),
):
    """Create a new conversation session."""
    conversation = ChatHistoryService.create_conversation(db, title=payload.title)
    return conversation


@router.get("", response_model=List[ConversationResponse])
def list_conversations(db: Session = Depends(get_db)):
    """List all conversations ordered by newest activity first."""
    return ChatHistoryService.list_conversations(db)


@router.get("/{conversation_id}", response_model=ConversationResponse)
def get_conversation(conversation_id: str, db: Session = Depends(get_db)):
    """Get single conversation metadata."""
    conversation = ChatHistoryService.get_conversation(db, conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    return conversation


@router.patch("/{conversation_id}", response_model=ConversationResponse)
def rename_conversation(
    conversation_id: str,
    payload: ConversationUpdate,
    db: Session = Depends(get_db),
):
    """Rename a conversation title."""
    conversation = ChatHistoryService.rename_conversation(
        db, conversation_id, new_title=payload.title
    )
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    return conversation


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: str, db: Session = Depends(get_db)):
    """Delete a conversation and all of its associated messages."""
    deleted = ChatHistoryService.delete_conversation(db, conversation_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    return None


@router.get("/{conversation_id}/messages", response_model=List[MessageResponse])
def get_messages(conversation_id: str, db: Session = Depends(get_db)):
    """Retrieve all messages in a conversation in chronological order."""
    conversation = ChatHistoryService.get_conversation(db, conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    return ChatHistoryService.get_messages(db, conversation_id)


@router.post("/{conversation_id}/messages", response_model=SendMessageResponse)
async def send_message(
    conversation_id: str,
    payload: MessageCreate,
    db: Session = Depends(get_db),
):
    """
    Non-streaming endpoint: Save user message, call AIService to generate response,
    and persist both messages in the conversation.
    """
    user_text = payload.message.strip()
    if not user_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty",
        )

    conversation = ChatHistoryService.get_conversation(db, conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    try:
        reply_text = await ai_service.chat(user_text)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI service temporarily unavailable. Please try again.",
        )

    user_msg, assistant_msg = ChatHistoryService.add_user_message_and_reply(
        db=db,
        conversation_id=conversation_id,
        user_content=user_text,
        reply_content=reply_text,
    )

    return SendMessageResponse(
        user_message=MessageResponse.model_validate(user_msg),
        assistant_message=MessageResponse.model_validate(assistant_msg),
    )


@router.post("/{conversation_id}/messages/stream")
async def send_message_stream(
    conversation_id: str,
    payload: MessageCreate,
    db: Session = Depends(get_db),
):
    """
    Streaming endpoint: Progressively yields AI response chunks via Server-Sent Events (SSE).
    Persists user message upfront, streams chunks, and persists assistant message upon completion.
    """
    user_text = payload.message.strip()
    if not user_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty",
        )

    conversation = ChatHistoryService.get_conversation(db, conversation_id)
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    # Persist user message immediately in a quick transaction
    user_msg = ChatHistoryService.add_user_message(
        db=db,
        conversation_id=conversation_id,
        user_content=user_text,
    )
    user_msg_data = MessageResponse.model_validate(user_msg).model_dump(mode="json")

    async def event_generator():
        # First event: inform client of the created user message
        yield f"event: user_message\ndata: {json.dumps(user_msg_data)}\n\n"

        accumulated_text = ""
        try:
            async for chunk in ai_service.chat_stream(user_text):
                accumulated_text += chunk
                chunk_data = json.dumps({"chunk": chunk})
                yield f"event: chunk\ndata: {chunk_data}\n\n"

            # Generation succeeded: persist the completed assistant message
            db_final = SessionLocal()
            try:
                assistant_msg = ChatHistoryService.add_assistant_message(
                    db=db_final,
                    conversation_id=conversation_id,
                    reply_content=accumulated_text,
                )
                assistant_msg_data = MessageResponse.model_validate(assistant_msg).model_dump(mode="json")
                yield f"event: complete\ndata: {json.dumps(assistant_msg_data)}\n\n"
            finally:
                db_final.close()

        except Exception:
            # Do not persist partial/failed answers; notify client with error event
            error_data = json.dumps({
                "error": "The AI service encountered an error while generating the response."
            })
            yield f"event: error\ndata: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
