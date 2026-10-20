'use client';

import React, { useState } from 'react';
import { ChatMessage as ChatMessageType } from '../types/chat';
import { cn, formatTime } from '@/lib/utils';

interface ChatMessageProps {
  message: ChatMessageType;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const [copied, setCopied] = useState<boolean>(false);
  const isUser = message.role === 'user';
  const isError = Boolean(message.isError);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard write fallback or silent fail
    }
  };

  return (
    <article
      className={cn(
        'group flex w-full gap-3 py-2 animate-fadeIn',
        isUser ? 'justify-end' : 'justify-start'
      )}
      aria-label={`${isUser ? 'User message' : 'Assistant response'}`}
    >
      {/* Bot Avatar (only on assistant messages) */}
      {!isUser && (
        <div
          className={cn(
            'flex h-8 w-8 shrink-0 select-none items-center justify-center rounded-full text-white shadow-xs ring-2',
            isError
              ? 'bg-red-500 ring-red-400/20'
              : 'bg-gradient-to-tr from-blue-600 via-indigo-600 to-violet-500 ring-blue-500/20'
          )}
          aria-hidden="true"
        >
          {isError ? (
            <svg
              className="h-4 w-4"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
          ) : (
            <svg
              className="h-4 w-4"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z" />
            </svg>
          )}
        </div>
      )}

      {/* Message Bubble Container */}
      <div
        className={cn(
          'flex max-w-[85%] flex-col sm:max-w-[75%]',
          isUser ? 'items-end' : 'items-start'
        )}
      >
        <div
          className={cn(
            'relative whitespace-pre-wrap rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-xs transition-shadow',
            isUser
              ? 'rounded-tr-xs bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-blue-500/10'
              : isError
              ? 'rounded-tl-xs border border-red-200 bg-red-50 text-red-900 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-200'
              : 'rounded-tl-xs border border-zinc-200/80 bg-white text-zinc-900 shadow-zinc-200/20 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-100'
          )}
        >
          {message.content}
          {message.isStreaming && !message.content && (
            <span className="flex items-center gap-1.5 py-0.5 text-xs text-zinc-400 dark:text-zinc-500">
              <span className="h-1.5 w-1.5 rounded-full bg-blue-500 animate-bounce" />
              <span
                className="h-1.5 w-1.5 rounded-full bg-indigo-500 animate-bounce"
                style={{ animationDelay: '150ms' }}
              />
              <span
                className="h-1.5 w-1.5 rounded-full bg-violet-500 animate-bounce"
                style={{ animationDelay: '300ms' }}
              />
            </span>
          )}
          {message.isStreaming && message.content && (
            <span
              className="inline-block w-1.5 h-3.5 ml-1 align-middle bg-blue-500 animate-pulse rounded-xs"
              aria-hidden="true"
            />
          )}
        </div>

        {/* Message Meta & Action Bar */}
        <div
          className={cn(
            'mt-1 flex items-center gap-2 px-1 text-[11px] text-zinc-400 dark:text-zinc-500'
          )}
        >
          <time dateTime={message.createdAt ? new Date(message.createdAt).toISOString() : undefined}>
            {formatTime(message.createdAt)}
          </time>

          {!isUser && !isError && (
            <button
              type="button"
              onClick={handleCopy}
              className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-zinc-500 hover:bg-zinc-100 hover:text-zinc-800 dark:text-zinc-400 dark:hover:bg-zinc-800 dark:hover:text-zinc-200 focus:outline-none focus:ring-1 focus:ring-blue-500"
              aria-label={copied ? 'Copied to clipboard' : 'Copy message text'}
            >
              {copied ? (
                <>
                  <svg
                    className="h-3 w-3 text-emerald-500"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    aria-hidden="true"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M5 13l4 4L19 7"
                    />
                  </svg>
                  <span className="text-[10px] text-emerald-600 dark:text-emerald-400 font-medium">
                    Copied
                  </span>
                </>
              ) : (
                <>
                  <svg
                    className="h-3 w-3"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    aria-hidden="true"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
                    />
                  </svg>
                  <span>Copy</span>
                </>
              )}
            </button>
          )}
        </div>
      </div>

      {/* User Avatar (only on user messages) */}
      {isUser && (
        <div
          className="flex h-8 w-8 shrink-0 select-none items-center justify-center rounded-full bg-zinc-700 text-white shadow-xs ring-2 ring-zinc-700/20 dark:bg-zinc-600"
          aria-hidden="true"
        >
          <svg
            className="h-4 w-4"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" />
            <circle cx="12" cy="7" r="4" />
          </svg>
        </div>
      )}
    </article>
  );
}
