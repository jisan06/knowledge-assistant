'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import {
  ChatMessage,
  ChatApiError,
  Conversation,
  MessageApiItem,
} from '../types/chat';
import { chatApi } from '../services/chat-api';
import { generateId } from '@/lib/utils';

const STORAGE_ACTIVE_CONVERSATION_KEY = 'ai_chat_active_conversation_id';

export interface UseChatReturn {
  conversations: Conversation[];
  currentConversationId: string | null;
  currentConversation: Conversation | null;
  messages: ChatMessage[];
  input: string;
  setInput: (value: string) => void;
  isLoading: boolean;
  isConversationsLoading: boolean;
  isMessagesLoading: boolean;
  error: string | null;
  conversationsError: string | null;
  sendMessage: (customMessage?: string) => Promise<void>;
  selectConversation: (conversationId: string) => Promise<void>;
  startNewChat: () => void;
  deleteConversation: (conversationId: string) => Promise<void>;
  clearChat: () => void;
  clearError: () => void;
  loadConversations: () => Promise<void>;
}

export function useChat(): UseChatReturn {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isConversationsLoading, setIsConversationsLoading] = useState<boolean>(true);
  const [isMessagesLoading, setIsMessagesLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [conversationsError, setConversationsError] = useState<string | null>(null);

  const isSubmittingRef = useRef<boolean>(false);
  const isInitialMountRef = useRef<boolean>(true);
  const activeAbortControllerRef = useRef<AbortController | null>(null);
  const activeConversationIdRef = useRef<string | null>(null);

  // Keep ref synchronized with currentConversationId state
  useEffect(() => {
    activeConversationIdRef.current = currentConversationId;
  }, [currentConversationId]);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const clearChat = useCallback(() => {
    setMessages([]);
    setError(null);
  }, []);

  /**
   * Fetch conversation history list from backend.
   */
  const loadConversations = useCallback(async () => {
    setIsConversationsLoading(true);
    setConversationsError(null);
    try {
      const list = await chatApi.getConversations();
      setConversations(list);
      return list;
    } catch (err: unknown) {
      const apiErr = err as ChatApiError;
      setConversationsError(apiErr.message || 'Failed to load conversation history.');
      return [];
    } finally {
      setIsConversationsLoading(false);
    }
  }, []);

  /**
   * Select and load an existing conversation session.
   */
  const selectConversation = useCallback(
    async (conversationId: string) => {
      if (currentConversationId === conversationId && !isMessagesLoading) {
        return;
      }

      // Abort any ongoing stream to avoid conversation cross-talk
      if (activeAbortControllerRef.current) {
        activeAbortControllerRef.current.abort();
        activeAbortControllerRef.current = null;
      }
      isSubmittingRef.current = false;
      setIsLoading(false);

      setCurrentConversationId(conversationId);
      activeConversationIdRef.current = conversationId;

      if (typeof window !== 'undefined') {
        localStorage.setItem(STORAGE_ACTIVE_CONVERSATION_KEY, conversationId);
      }

      setError(null);
      setIsMessagesLoading(true);

      try {
        const loadedMessages = await chatApi.getConversationMessages(conversationId);
        setMessages(loadedMessages);
      } catch (err: unknown) {
        const apiErr = err as ChatApiError;
        setError(apiErr.message || 'Failed to load messages for this conversation.');
        setMessages([]);
      } finally {
        setIsMessagesLoading(false);
      }
    },
    [currentConversationId, isMessagesLoading]
  );

  /**
   * Start a brand new chat (resets active conversation).
   */
  const startNewChat = useCallback(() => {
    // Abort any ongoing stream
    if (activeAbortControllerRef.current) {
      activeAbortControllerRef.current.abort();
      activeAbortControllerRef.current = null;
    }
    isSubmittingRef.current = false;
    setIsLoading(false);

    setCurrentConversationId(null);
    activeConversationIdRef.current = null;
    setMessages([]);
    setError(null);
    setInput('');
    if (typeof window !== 'undefined') {
      localStorage.removeItem(STORAGE_ACTIVE_CONVERSATION_KEY);
    }
  }, []);

  /**
   * Delete a conversation.
   */
  const deleteConversation = useCallback(
    async (conversationId: string) => {
      try {
        await chatApi.deleteConversation(conversationId);
        setConversations((prev) => prev.filter((c) => c.id !== conversationId));

        if (currentConversationId === conversationId) {
          startNewChat();
        }
      } catch (err: unknown) {
        const apiErr = err as ChatApiError;
        setError(apiErr.message || 'Failed to delete conversation.');
      }
    },
    [currentConversationId, startNewChat]
  );

  /**
   * Initial load: fetch conversations and restore last active conversation if present.
   */
  useEffect(() => {
    if (!isInitialMountRef.current) return;
    isInitialMountRef.current = false;

    let isSubscribed = true;

    async function init() {
      const list = await loadConversations();
      if (!isSubscribed) return;

      if (typeof window !== 'undefined') {
        const savedActiveId = localStorage.getItem(STORAGE_ACTIVE_CONVERSATION_KEY);
        if (savedActiveId && list.some((c) => c.id === savedActiveId)) {
          selectConversation(savedActiveId);
        } else if (list.length > 0) {
          selectConversation(list[0].id);
        }
      }
    }

    init();

    return () => {
      isSubscribed = false;
    };
  }, [loadConversations, selectConversation]);

  /**
   * Send a message to the active conversation with progressive streaming response.
   */
  const sendMessage = useCallback(
    async (customMessage?: string): Promise<void> => {
      const messageText = (customMessage !== undefined ? customMessage : input).trim();

      if (!messageText || isSubmittingRef.current) {
        return;
      }

      isSubmittingRef.current = true;
      setIsLoading(true);
      setError(null);

      if (customMessage === undefined) {
        setInput('');
      }

      const tempUserMsgId = generateId();
      const tempAssistantMsgId = generateId();

      // Optimistic user message
      const tempUserMessage: ChatMessage = {
        id: tempUserMsgId,
        role: 'user',
        content: messageText,
        createdAt: new Date(),
      };

      // Optimistic assistant streaming placeholder
      const tempAssistantMessage: ChatMessage = {
        id: tempAssistantMsgId,
        role: 'assistant',
        content: '',
        createdAt: new Date(),
        isStreaming: true,
      };

      setMessages((prev) => [...prev, tempUserMessage, tempAssistantMessage]);

      // Set up abort controller for client cancellation
      const abortController = new AbortController();
      activeAbortControllerRef.current = abortController;

      try {
        let targetConversationId = currentConversationId;

        // If in New Chat mode, create persistent conversation session first
        if (!targetConversationId) {
          const newConversation = await chatApi.createConversation();
          targetConversationId = newConversation.id;
          setCurrentConversationId(newConversation.id);
          activeConversationIdRef.current = newConversation.id;

          if (typeof window !== 'undefined') {
            localStorage.setItem(STORAGE_ACTIVE_CONVERSATION_KEY, newConversation.id);
          }
        }

        // Stream AI response chunks via SSE
        await chatApi.streamMessageToConversation(
          targetConversationId,
          messageText,
          {
            onUserMessage: (persistedUserMsg: MessageApiItem) => {
              if (activeConversationIdRef.current !== targetConversationId) return;
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === tempUserMsgId
                    ? {
                        ...m,
                        id: persistedUserMsg.id,
                        conversationId: persistedUserMsg.conversation_id,
                        content: persistedUserMsg.content,
                      }
                    : m
                )
              );
            },
            onChunk: (chunk: string) => {
              if (activeConversationIdRef.current !== targetConversationId) return;
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === tempAssistantMsgId
                    ? {
                        ...m,
                        content: m.content + chunk,
                        isStreaming: true,
                      }
                    : m
                )
              );
            },
            onComplete: (persistedAssistantMsg: MessageApiItem) => {
              if (activeConversationIdRef.current !== targetConversationId) return;
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === tempAssistantMsgId
                    ? {
                        ...m,
                        id: persistedAssistantMsg.id,
                        conversationId: persistedAssistantMsg.conversation_id,
                        content: persistedAssistantMsg.content,
                        isStreaming: false,
                      }
                    : m
                )
              );
              setIsLoading(false);
              isSubmittingRef.current = false;
              activeAbortControllerRef.current = null;
              loadConversations();
            },
            onError: (errorMessage: string) => {
              if (activeConversationIdRef.current !== targetConversationId) return;
              setError(errorMessage);
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === tempAssistantMsgId
                    ? {
                        ...m,
                        content: m.content || errorMessage,
                        isError: true,
                        isStreaming: false,
                      }
                    : m
                )
              );
              setIsLoading(false);
              isSubmittingRef.current = false;
              activeAbortControllerRef.current = null;
            },
          },
          abortController.signal
        );
      } catch (err: unknown) {
        const apiError = err as ChatApiError;
        const errorMessage = apiError.message || 'An unexpected error occurred while streaming.';
        setError(errorMessage);

        setMessages((prev) =>
          prev.map((m) =>
            m.id === tempAssistantMsgId
              ? {
                  ...m,
                  content: m.content || errorMessage,
                  isError: true,
                  isStreaming: false,
                }
              : m
          )
        );
        setIsLoading(false);
        isSubmittingRef.current = false;
        activeAbortControllerRef.current = null;
      }
    },
    [currentConversationId, input, loadConversations]
  );

  const currentConversation =
    conversations.find((c) => c.id === currentConversationId) || null;

  return {
    conversations,
    currentConversationId,
    currentConversation,
    messages,
    input,
    setInput,
    isLoading,
    isConversationsLoading,
    isMessagesLoading,
    error,
    conversationsError,
    sendMessage,
    selectConversation,
    startNewChat,
    deleteConversation,
    clearChat,
    clearError,
    loadConversations,
  };
}
