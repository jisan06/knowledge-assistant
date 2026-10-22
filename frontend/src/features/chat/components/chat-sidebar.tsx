'use client';

import React from 'react';
import Link from 'next/link';
import { Conversation } from '../types/chat';
import { Button } from '@/components/ui/button';
import { cn, formatTime } from '@/lib/utils';

interface ChatSidebarProps {
  conversations: Conversation[];
  currentConversationId: string | null;
  isLoading: boolean;
  error: string | null;
  isOpen: boolean;
  onClose: () => void;
  onSelectConversation: (id: string) => void;
  onNewChat: () => void;
  onDeleteConversation: (id: string) => void;
  onRetryLoad: () => void;
}

export function ChatSidebar({
  conversations,
  currentConversationId,
  isLoading,
  error,
  isOpen,
  onClose,
  onSelectConversation,
  onNewChat,
  onDeleteConversation,
  onRetryLoad,
}: ChatSidebarProps) {
  const handleDelete = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (confirm('Delete this conversation?')) {
      onDeleteConversation(id);
    }
  };

  const sidebarContent = (
    <aside
      aria-label="Conversation history"
      className="flex h-full w-72 flex-col border-r border-zinc-200/80 bg-zinc-50/95 p-3 dark:border-zinc-800 dark:bg-zinc-900/95"
    >
      {/* Sidebar Header & New Chat Button */}
      <div className="pb-3 border-b border-zinc-200/60 dark:border-zinc-800/80">
        <Button
          type="button"
          onClick={() => {
            onNewChat();
            onClose();
          }}
          className="w-full justify-start gap-2.5 rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-medium shadow-xs"
        >
          <svg
            className="h-4 w-4"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
          </svg>
          <span>New Chat</span>
        </Button>
      </div>

      {/* Conversations List */}
      <div className="flex-1 overflow-y-auto py-3 space-y-1">
        <div className="px-2 pb-1.5 text-[11px] font-semibold tracking-wider text-zinc-400 uppercase dark:text-zinc-500">
          Recent Conversations
        </div>

        {/* Loading Skeleton */}
        {isLoading && (
          <div className="space-y-2 p-2">
            {[1, 2, 3, 4].map((i) => (
              <div
                key={i}
                className="h-10 w-full animate-pulse rounded-xl bg-zinc-200/70 dark:bg-zinc-800/70"
              />
            ))}
          </div>
        )}

        {/* Error State */}
        {!isLoading && error && (
          <div className="rounded-xl border border-red-200/80 bg-red-50/60 p-3 text-center text-xs text-red-700 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-300">
            <p className="mb-2">{error}</p>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={onRetryLoad}
              className="text-xs"
            >
              Retry
            </Button>
          </div>
        )}

        {/* Empty State */}
        {!isLoading && !error && conversations.length === 0 && (
          <div className="flex flex-col items-center justify-center p-6 text-center text-xs text-zinc-400 dark:text-zinc-500">
            <svg
              className="h-8 w-8 mb-2 opacity-50"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.5}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
              />
            </svg>
            <p>No chat history yet.</p>
            <p className="mt-1 text-[11px]">Start a conversation above.</p>
          </div>
        )}

        {/* Conversation Items */}
        {!isLoading &&
          !error &&
          conversations.map((conv) => {
            const isSelected = conv.id === currentConversationId;
            return (
              <div
                key={conv.id}
                role="button"
                tabIndex={0}
                onClick={() => {
                  onSelectConversation(conv.id);
                  onClose();
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    onSelectConversation(conv.id);
                    onClose();
                  }
                }}
                className={cn(
                  'group relative flex w-full cursor-pointer items-center justify-between rounded-xl px-3 py-2.5 text-xs transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500',
                  isSelected
                    ? 'bg-blue-50 text-blue-900 font-medium dark:bg-blue-950/60 dark:text-blue-200'
                    : 'text-zinc-700 hover:bg-zinc-100 dark:text-zinc-300 dark:hover:bg-zinc-800/80'
                )}
              >
                <div className="flex flex-col min-w-0 pr-2">
                  <span className="truncate leading-snug">{conv.title}</span>
                  <span className="text-[10px] text-zinc-400 dark:text-zinc-500 mt-0.5">
                    {formatTime(conv.updatedAt)}
                  </span>
                </div>

                <button
                  type="button"
                  title="Delete conversation"
                  aria-label={`Delete conversation ${conv.title}`}
                  onClick={(e) => handleDelete(e, conv.id)}
                  className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-zinc-200 text-zinc-400 hover:text-red-600 dark:hover:bg-zinc-700 dark:hover:text-red-400 transition-opacity"
                >
                  <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                    />
                  </svg>
                </button>
              </div>
            );
          })}
      </div>

      {/* Sidebar Navigation Footer */}
      <div className="pt-2 mt-auto border-t border-zinc-200/60 dark:border-zinc-800/80">
        <Link
          href="/documents"
          onClick={onClose}
          className="flex items-center gap-2 rounded-xl px-3 py-2 text-xs font-medium text-zinc-600 hover:bg-zinc-100 hover:text-zinc-900 dark:text-zinc-400 dark:hover:bg-zinc-800/80 dark:hover:text-zinc-200 transition-colors"
        >
          <svg
            className="h-4 w-4 shrink-0 text-violet-600 dark:text-violet-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
            aria-hidden="true"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z"
            />
          </svg>
          <span className="truncate">Documents & PDF Upload</span>
        </Link>
      </div>
    </aside>
  );

  return (
    <>
      {/* Desktop Sidebar (Permanent) */}
      <div className="hidden md:flex h-full shrink-0">{sidebarContent}</div>

      {/* Mobile Drawer (Collapsible) */}
      {isOpen && (
        <div className="fixed inset-0 z-50 flex md:hidden animate-fadeIn">
          {/* Backdrop overlay */}
          <div
            className="fixed inset-0 bg-black/40 backdrop-blur-xs transition-opacity"
            onClick={onClose}
            aria-hidden="true"
          />

          {/* Drawer container */}
          <div className="relative flex w-72 max-w-[80vw] flex-1 flex-col shadow-xl">
            {sidebarContent}
          </div>
        </div>
      )}
    </>
  );
}
