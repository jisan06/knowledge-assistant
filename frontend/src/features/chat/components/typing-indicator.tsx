'use client';

import React from 'react';

export function TypingIndicator() {
  return (
    <div
      role="status"
      aria-label="AI is typing"
      className="flex items-start gap-3 py-1 animate-fadeIn"
    >
      {/* Bot Avatar Icon */}
      <div className="flex h-8 w-8 shrink-0 select-none items-center justify-center rounded-full bg-gradient-to-tr from-blue-600 via-indigo-600 to-violet-500 text-white shadow-sm ring-2 ring-blue-500/20">
        <svg
          className="h-4 w-4"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z" />
        </svg>
      </div>

      {/* Typing Bubble */}
      <div className="flex items-center gap-1.5 rounded-2xl rounded-tl-sm border border-zinc-200/80 bg-white px-4 py-3 shadow-xs dark:border-zinc-800 dark:bg-zinc-900">
        <span className="text-xs font-medium text-zinc-500 dark:text-zinc-400 mr-1">
          AI is thinking
        </span>
        <span
          className="h-2 w-2 rounded-full bg-blue-500 animate-bounce"
          style={{ animationDuration: '1s', animationDelay: '0ms' }}
        />
        <span
          className="h-2 w-2 rounded-full bg-indigo-500 animate-bounce"
          style={{ animationDuration: '1s', animationDelay: '150ms' }}
        />
        <span
          className="h-2 w-2 rounded-full bg-violet-500 animate-bounce"
          style={{ animationDuration: '1s', animationDelay: '300ms' }}
        />
      </div>
    </div>
  );
}
