/**
 * Public exports for the Chat Feature module
 */

export { ChatInterface } from './components/chat-interface';
export { ChatHeader } from './components/chat-header';
export { ChatSidebar } from './components/chat-sidebar';
export { ChatMessage } from './components/chat-message';
export { ChatInput } from './components/chat-input';
export { ChatEmptyState } from './components/chat-empty-state';
export { TypingIndicator } from './components/typing-indicator';

export { useChat } from './hooks/use-chat';
export { chatApi } from './services/chat-api';
export * from './types/chat';
