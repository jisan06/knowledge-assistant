from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict


class MessageBase(BaseModel):
    role: str
    content: str


class MessageCreate(BaseModel):
    message: str = Field(..., min_length=1, description="Message text from user")


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ConversationCreate(BaseModel):
    title: Optional[str] = Field(None, max_length=255, description="Optional title")


class ConversationUpdate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255, description="New title")


class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SendMessageResponse(BaseModel):
    user_message: MessageResponse
    assistant_message: MessageResponse


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    reply: str
