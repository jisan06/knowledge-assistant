/**
 * Chat Feature Type Definitions
 */

export type MessageRole = 'user' | 'assistant' | 'system';

export interface ChatMessage {
  id: string;
  conversationId?: string;
  role: MessageRole;
  content: string;
  createdAt: Date;
  isError?: boolean;
  isStreaming?: boolean;
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface ConversationApiItem {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface MessageApiItem {
  id: string;
  conversation_id: string;
  role: string;
  content: string;
  created_at: string;
}

export interface SendMessageApiResponse {
  user_message: MessageApiItem;
  assistant_message: MessageApiItem;
}

export interface ChatRequest {
  message: string;
}

export interface ChatResponse {
  reply: string;
}

export interface ChatApiError {
  message: string;
  statusCode?: number;
}

export interface ChatState {
  conversations: Conversation[];
  currentConversationId: string | null;
  messages: ChatMessage[];
  isLoading: boolean;
  isConversationsLoading: boolean;
  error: string | null;
}
