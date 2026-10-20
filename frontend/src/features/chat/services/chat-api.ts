import {
  ChatRequest,
  ChatResponse,
  ChatApiError,
  Conversation,
  ChatMessage,
  ConversationApiItem,
  MessageApiItem,
  SendMessageApiResponse,
} from '../types/chat';

export interface StreamMessageCallbacks {
  onUserMessage?: (userMsg: MessageApiItem) => void;
  onChunk: (chunk: string) => void;
  onComplete: (assistantMsg: MessageApiItem) => void;
  onError: (errorMessage: string) => void;
}

/**
 * Service to manage all HTTP interactions with the AI Chat Backend.
 */
class ChatApiService {
  private get baseUrl(): string {
    return process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000';
  }

  private get cleanBaseUrl(): string {
    return this.baseUrl.replace(/\/+$/, '');
  }

  private formatError(response: Response, defaultMessage: string): ChatApiError {
    let userFriendlyMessage = defaultMessage;

    if (response.status === 400) {
      userFriendlyMessage = 'Invalid request. Please check your message.';
    } else if (response.status === 404) {
      userFriendlyMessage = 'Conversation or resource not found.';
    } else if (response.status === 429) {
      userFriendlyMessage = 'Too many requests. Please wait a moment before trying again.';
    } else if (response.status === 502 || response.status === 503) {
      userFriendlyMessage = 'The AI service is temporarily unavailable. Please try again shortly.';
    } else if (response.status >= 500) {
      userFriendlyMessage = 'The server encountered an error. Please try again later.';
    }

    return {
      message: userFriendlyMessage,
      statusCode: response.status,
    };
  }

  /**
   * Fetch all persistent conversations ordered by newest activity.
   */
  async getConversations(): Promise<Conversation[]> {
    try {
      const response = await fetch(`${this.cleanBaseUrl}/api/conversations`, {
        method: 'GET',
        headers: { Accept: 'application/json' },
      });

      if (!response.ok) {
        throw this.formatError(response, 'Failed to fetch conversation history.');
      }

      const data: ConversationApiItem[] = await response.json();
      return data.map((item) => ({
        id: item.id,
        title: item.title,
        createdAt: new Date(item.created_at),
        updatedAt: new Date(item.updated_at),
      }));
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'statusCode' in err && 'message' in err) {
        throw err;
      }
      throw {
        message: 'Unable to reach the server to load conversations.',
      } as ChatApiError;
    }
  }

  /**
   * Create a new conversation session.
   */
  async createConversation(title?: string): Promise<Conversation> {
    try {
      const response = await fetch(`${this.cleanBaseUrl}/api/conversations`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
        body: JSON.stringify({ title }),
      });

      if (!response.ok) {
        throw this.formatError(response, 'Failed to create a new conversation.');
      }

      const item: ConversationApiItem = await response.json();
      return {
        id: item.id,
        title: item.title,
        createdAt: new Date(item.created_at),
        updatedAt: new Date(item.updated_at),
      };
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'statusCode' in err && 'message' in err) {
        throw err;
      }
      throw {
        message: 'Unable to create new conversation.',
      } as ChatApiError;
    }
  }

  /**
   * Retrieve all messages in a conversation in chronological order.
   */
  async getConversationMessages(conversationId: string): Promise<ChatMessage[]> {
    try {
      const response = await fetch(
        `${this.cleanBaseUrl}/api/conversations/${encodeURIComponent(conversationId)}/messages`,
        {
          method: 'GET',
          headers: { Accept: 'application/json' },
        }
      );

      if (!response.ok) {
        throw this.formatError(response, 'Failed to retrieve messages.');
      }

      const data: MessageApiItem[] = await response.json();
      return data.map((item) => ({
        id: item.id,
        conversationId: item.conversation_id,
        role: item.role as 'user' | 'assistant',
        content: item.content,
        createdAt: new Date(item.created_at),
      }));
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'statusCode' in err && 'message' in err) {
        throw err;
      }
      throw {
        message: 'Unable to load conversation messages.',
      } as ChatApiError;
    }
  }

  /**
   * Non-streaming: Send a message to an existing conversation and persist user message & assistant reply.
   */
  async sendMessageToConversation(
    conversationId: string,
    message: string,
    signal?: AbortSignal
  ): Promise<SendMessageApiResponse> {
    const trimmed = message.trim();
    if (!trimmed) {
      throw {
        message: 'Message cannot be empty.',
        statusCode: 400,
      } as ChatApiError;
    }

    try {
      const response = await fetch(
        `${this.cleanBaseUrl}/api/conversations/${encodeURIComponent(conversationId)}/messages`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'application/json',
          },
          body: JSON.stringify({ message: trimmed }),
          signal,
        }
      );

      if (!response.ok) {
        throw this.formatError(response, 'Failed to send message.');
      }

      return await response.json();
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'statusCode' in err && 'message' in err) {
        throw err;
      }
      if (err instanceof DOMException && err.name === 'AbortError') {
        throw {
          message: 'Request was cancelled.',
        } as ChatApiError;
      }
      throw {
        message: 'Unable to connect to the backend server. Please verify the service is running.',
      } as ChatApiError;
    }
  }

  /**
   * Streaming: Send a message and stream assistant reply chunks using Server-Sent Events (SSE).
   */
  async streamMessageToConversation(
    conversationId: string,
    message: string,
    callbacks: StreamMessageCallbacks,
    signal?: AbortSignal
  ): Promise<void> {
    const trimmed = message.trim();
    if (!trimmed) {
      throw {
        message: 'Message cannot be empty.',
        statusCode: 400,
      } as ChatApiError;
    }

    let response: Response;
    try {
      response = await fetch(
        `${this.cleanBaseUrl}/api/conversations/${encodeURIComponent(conversationId)}/messages/stream`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'text/event-stream',
          },
          body: JSON.stringify({ message: trimmed }),
          signal,
        }
      );
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        return;
      }
      callbacks.onError('Unable to connect to the backend server. Please verify the service is running.');
      return;
    }

    if (!response.ok) {
      let errorMsg = 'Failed to stream response.';
      try {
        const errJson = await response.json();
        if (errJson.detail) errorMsg = errJson.detail;
      } catch {
        // Fallback to formatted error
        errorMsg = this.formatError(response, 'Failed to stream response.').message;
      }
      callbacks.onError(errorMsg);
      return;
    }

    if (!response.body) {
      callbacks.onError('Streaming response body is unavailable.');
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop() || '';

        for (const part of parts) {
          const lines = part.trim().split('\n');
          let eventType = 'message';
          let dataStr = '';

          for (const line of lines) {
            if (line.startsWith('event:')) {
              eventType = line.slice(6).trim();
            } else if (line.startsWith('data:')) {
              dataStr += line.slice(5).trim();
            }
          }

          if (!dataStr) continue;

          try {
            const parsed = JSON.parse(dataStr);
            if (eventType === 'user_message') {
              callbacks.onUserMessage?.(parsed as MessageApiItem);
            } else if (eventType === 'chunk') {
              if (parsed.chunk) {
                callbacks.onChunk(parsed.chunk);
              }
            } else if (eventType === 'complete') {
              callbacks.onComplete(parsed as MessageApiItem);
            } else if (eventType === 'error') {
              callbacks.onError(parsed.error || 'An error occurred during streaming.');
            }
          } catch {
            // Skip invalid event chunks
          }
        }
      }
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        return;
      }
      callbacks.onError('Connection was interrupted while streaming AI response.');
    }
  }

  /**
   * Delete a conversation by ID.
   */
  async deleteConversation(conversationId: string): Promise<void> {
    try {
      const response = await fetch(
        `${this.cleanBaseUrl}/api/conversations/${encodeURIComponent(conversationId)}`,
        {
          method: 'DELETE',
        }
      );

      if (!response.ok && response.status !== 404) {
        throw this.formatError(response, 'Failed to delete conversation.');
      }
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'statusCode' in err && 'message' in err) {
        throw err;
      }
      throw {
        message: 'Unable to delete conversation.',
      } as ChatApiError;
    }
  }

  /**
   * Legacy non-streaming endpoint: Sends message directly to POST /api/chat.
   */
  async sendMessage(message: string, signal?: AbortSignal): Promise<ChatResponse> {
    const trimmed = message.trim();
    if (!trimmed) {
      throw {
        message: 'Message cannot be empty.',
        statusCode: 400,
      } as ChatApiError;
    }

    try {
      const response = await fetch(`${this.cleanBaseUrl}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
        body: JSON.stringify({ message: trimmed } as ChatRequest),
        signal,
      });

      if (!response.ok) {
        throw this.formatError(response, 'Failed to get an AI response.');
      }

      return await response.json();
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'statusCode' in err && 'message' in err) {
        throw err;
      }
      if (err instanceof DOMException && err.name === 'AbortError') {
        throw { message: 'Request was cancelled.' } as ChatApiError;
      }
      throw {
        message: 'Unable to reach the AI chat server.',
      } as ChatApiError;
    }
  }
}

export const chatApi = new ChatApiService();
