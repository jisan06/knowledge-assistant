'use client';

import React from 'react';

interface ChatEmptyStateProps {
  onSelectPrompt: (promptText: string) => void;
}

const SUGGESTED_PROMPTS = [
  {
    title: 'Explain a Concept',
    description: 'Explain quantum computing in simple terms for a beginner.',
    prompt: 'Explain quantum computing in simple terms for a beginner.',
    icon: (
      <svg className="h-5 w-5 text-amber-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
    ),
  },
  {
    title: 'Code Assistance',
    description: 'Write a Python FastAPI snippet to validate request schemas.',
    prompt: 'Write a Python FastAPI snippet to validate request schemas using Pydantic.',
    icon: (
      <svg className="h-5 w-5 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
      </svg>
    ),
  },
  {
    title: 'Productivity & Planning',
    description: 'Give me 5 actionable productivity habits for developers.',
    prompt: 'What are 5 actionable productivity habits for professional software engineers?',
    icon: (
      <svg className="h-5 w-5 text-emerald-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
      </svg>
    ),
  },
  {
    title: 'Professional Writing',
    description: 'Draft a friendly follow-up email to a project stakeholder.',
    prompt: 'Draft a polite and concise follow-up email to a client regarding project milestone review.',
    icon: (
      <svg className="h-5 w-5 text-purple-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
      </svg>
    ),
  },
];

export function ChatEmptyState({ onSelectPrompt }: ChatEmptyStateProps) {
  return (
    <div className="m-auto flex max-w-xl flex-col items-center justify-center px-4 py-8 text-center animate-fadeIn">
      {/* Decorative Hero Icon */}
      <div className="relative mb-5 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-tr from-blue-600 via-indigo-600 to-violet-500 text-white shadow-lg shadow-blue-500/20 ring-4 ring-blue-500/10 dark:shadow-blue-500/10">
        <svg
          className="h-8 w-8"
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

      {/* Heading & Subtitle */}
      <h2 className="text-xl font-bold tracking-tight text-zinc-900 dark:text-zinc-50 sm:text-2xl">
        How can I help you today?
      </h2>
      <p className="mt-2 text-sm text-zinc-500 dark:text-zinc-400 max-w-md">
        Start a conversation with AI Chat. Ask questions, brainstorm ideas, write code, or summarize information.
      </p>

      {/* Suggestion Cards Grid */}
      <div className="mt-8 grid w-full grid-cols-1 gap-3 sm:grid-cols-2 text-left">
        {SUGGESTED_PROMPTS.map((item, index) => (
          <button
            key={index}
            type="button"
            onClick={() => onSelectPrompt(item.prompt)}
            className="group flex flex-col justify-between rounded-xl border border-zinc-200/90 bg-white p-3.5 text-left shadow-xs transition-all hover:border-blue-400 hover:shadow-sm hover:-translate-y-0.5 dark:border-zinc-800 dark:bg-zinc-900 dark:hover:border-blue-500/60 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <div className="flex items-center gap-2.5">
              <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-zinc-100 group-hover:bg-blue-50 dark:bg-zinc-800 dark:group-hover:bg-blue-950/40 transition-colors">
                {item.icon}
              </span>
              <span className="text-xs font-semibold text-zinc-900 group-hover:text-blue-600 dark:text-zinc-200 dark:group-hover:text-blue-400 transition-colors">
                {item.title}
              </span>
            </div>
            <p className="mt-2 line-clamp-2 text-xs text-zinc-500 dark:text-zinc-400">
              {item.description}
            </p>
          </button>
        ))}
      </div>
    </div>
  );
}
