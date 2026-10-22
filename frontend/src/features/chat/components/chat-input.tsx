'use client';

import React, { useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface ChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  isLoading: boolean;
  placeholder?: string;
  isRagEnabled?: boolean;
  onToggleRag?: () => void;
}

export function ChatInput({
  value,
  onChange,
  onSubmit,
  isLoading,
  placeholder,
  isRagEnabled = false,
  onToggleRag,
}: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const defaultPlaceholder = isRagEnabled
    ? 'Ask a question about your uploaded documents... (Shift+Enter for new line)'
    : 'Ask AI Chat anything... (Shift+Enter for new line)';
  const activePlaceholder = placeholder || defaultPlaceholder;

  // Auto-resize textarea height based on content
  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    textarea.style.height = 'auto';
    const nextHeight = Math.min(textarea.scrollHeight, 180);
    textarea.style.height = `${Math.max(nextHeight, 48)}px`;
  }, [value]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Submit on Enter without Shift
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isLoading && value.trim()) {
        onSubmit();
      }
    }
  };

  const handleFormSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!isLoading && value.trim()) {
      onSubmit();
    }
  };

  const canSubmit = !isLoading && value.trim().length > 0;

  return (
    <div className="flex w-full flex-col gap-2">
      {/* Grounding Mode Toggle Bar */}
      {onToggleRag && (
        <div className="flex items-center justify-between px-1 text-xs">
          <button
            type="button"
            onClick={onToggleRag}
            aria-pressed={isRagEnabled}
            title={
              isRagEnabled
                ? 'Document grounding is ON: answers will retrieve and cite uploaded PDFs'
                : 'Document grounding is OFF: ordinary chat mode'
            }
            className={cn(
              'inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium transition-all focus:outline-none focus:ring-2 focus:ring-violet-500/40 border cursor-pointer',
              isRagEnabled
                ? 'bg-violet-50 text-violet-700 border-violet-200/90 dark:bg-violet-950/60 dark:text-violet-300 dark:border-violet-800 shadow-2xs'
                : 'bg-zinc-100/80 text-zinc-500 border-zinc-200 hover:bg-zinc-200/70 hover:text-zinc-700 dark:bg-zinc-800/80 dark:text-zinc-400 dark:border-zinc-700/80 dark:hover:bg-zinc-800'
            )}
          >
            <svg
              className={cn(
                'h-3.5 w-3.5 shrink-0 transition-colors',
                isRagEnabled ? 'text-violet-600 dark:text-violet-400' : 'text-zinc-400 dark:text-zinc-500'
              )}
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
            <span>Document Grounding: {isRagEnabled ? 'ON' : 'OFF'}</span>
          </button>

          {isRagEnabled && (
            <span className="text-[11px] text-violet-600 dark:text-violet-400 hidden sm:inline-flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-violet-500 animate-pulse" />
              Searching uploaded documents
            </span>
          )}
        </div>
      )}

      <form
        onSubmit={handleFormSubmit}
        className="relative flex w-full items-end gap-2 rounded-2xl border border-zinc-200/90 bg-white p-2 shadow-sm transition-all focus-within:border-blue-500 focus-within:ring-2 focus-within:ring-blue-500/20 dark:border-zinc-800 dark:bg-zinc-900"
      >
        <label htmlFor="chat-message-input" className="sr-only">
          Your message for AI Chat
        </label>

        <textarea
          id="chat-message-input"
          ref={textareaRef}
          rows={1}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={activePlaceholder}
          disabled={isLoading}
          className="min-h-[48px] max-h-[180px] w-full resize-none bg-transparent px-3 py-3 text-sm leading-relaxed text-zinc-900 placeholder-zinc-400 focus:outline-none disabled:cursor-not-allowed disabled:opacity-60 dark:text-zinc-100 dark:placeholder-zinc-500"
        />

      <div className="flex shrink-0 items-center pb-1">
        <Button
          type="submit"
          size="icon"
          disabled={!canSubmit}
          aria-label="Send message"
          className="h-10 w-10 rounded-xl bg-blue-600 hover:bg-blue-700 text-white disabled:bg-zinc-200 disabled:text-zinc-400 dark:disabled:bg-zinc-800 dark:disabled:text-zinc-600 transition-colors"
        >
          {isLoading ? (
            <svg
              className="h-4 w-4 animate-spin"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              />
            </svg>
          ) : (
            <svg
              className="h-4 w-4 translate-x-0.5"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          )}
        </Button>
      </div>
    </form>
  </div>
);
}
