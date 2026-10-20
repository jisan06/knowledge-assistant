from typing import Optional, List, Tuple
from sqlalchemy.orm import Session

from app.db.models import Conversation, Message, utc_now


def generate_title_from_message(message: str, max_length: int = 40) -> str:
    """Generate a clean, concise conversation title from the first message."""
    clean = " ".join(message.strip().split())
    if len(clean) <= max_length:
        return clean
    return clean[:max_length].rstrip() + "..."


class ChatHistoryService:
    @staticmethod
    def create_conversation(db: Session, title: Optional[str] = None) -> Conversation:
        conversation = Conversation(
            title=title.strip() if title and title.strip() else "New Chat"
        )
        try:
            db.add(conversation)
            db.commit()
            db.refresh(conversation)
            return conversation
        except Exception:
            db.rollback()
            raise

    @staticmethod
    def list_conversations(db: Session) -> List[Conversation]:
        return (
            db.query(Conversation)
            .order_by(Conversation.updated_at.desc())
            .all()
        )

    @staticmethod
    def get_conversation(db: Session, conversation_id: str) -> Optional[Conversation]:
        return (
            db.query(Conversation)
            .filter(Conversation.id == conversation_id)
            .first()
        )

    @staticmethod
    def get_messages(db: Session, conversation_id: str) -> List[Message]:
        return (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .all()
        )

    @staticmethod
    def rename_conversation(
        db: Session, conversation_id: str, new_title: str
    ) -> Optional[Conversation]:
        conversation = ChatHistoryService.get_conversation(db, conversation_id)
        if not conversation:
            return None

        try:
            conversation.title = new_title.strip()
            conversation.updated_at = utc_now()
            db.commit()
            db.refresh(conversation)
            return conversation
        except Exception:
            db.rollback()
            raise

    @staticmethod
    def delete_conversation(db: Session, conversation_id: str) -> bool:
        conversation = ChatHistoryService.get_conversation(db, conversation_id)
        if not conversation:
            return False

        try:
            db.delete(conversation)
            db.commit()
            return True
        except Exception:
            db.rollback()
            raise

    @staticmethod
    def add_user_message(
        db: Session,
        conversation_id: str,
        user_content: str,
    ) -> Message:
        conversation = ChatHistoryService.get_conversation(db, conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")

        # If conversation is titled "New Chat", auto-generate a title from the first message
        if conversation.title == "New Chat":
            conversation.title = generate_title_from_message(user_content)

        user_message = Message(
            conversation_id=conversation_id,
            role="user",
            content=user_content,
        )

        conversation.updated_at = utc_now()

        try:
            db.add(user_message)
            db.commit()
            db.refresh(user_message)
            return user_message
        except Exception:
            db.rollback()
            raise

    @staticmethod
    def add_assistant_message(
        db: Session,
        conversation_id: str,
        reply_content: str,
    ) -> Message:
        conversation = ChatHistoryService.get_conversation(db, conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")

        assistant_message = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=reply_content,
        )

        conversation.updated_at = utc_now()

        try:
            db.add(assistant_message)
            db.commit()
            db.refresh(assistant_message)
            return assistant_message
        except Exception:
            db.rollback()
            raise

    @staticmethod
    def add_user_message_and_reply(
        db: Session,
        conversation_id: str,
        user_content: str,
        reply_content: str,
    ) -> Tuple[Message, Message]:
        conversation = ChatHistoryService.get_conversation(db, conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")

        # If conversation is titled "New Chat", auto-generate a title from the first message
        if conversation.title == "New Chat":
            conversation.title = generate_title_from_message(user_content)

        user_message = Message(
            conversation_id=conversation_id,
            role="user",
            content=user_content,
        )

        assistant_message = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=reply_content,
        )

        now = utc_now()
        conversation.updated_at = now

        try:
            db.add(user_message)
            db.add(assistant_message)
            db.commit()
            db.refresh(user_message)
            db.refresh(assistant_message)
            return user_message, assistant_message
        except Exception:
            db.rollback()
            raise
