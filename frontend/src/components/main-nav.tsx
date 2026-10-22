'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';

interface MainNavProps {
  className?: string;
  activeSection?: 'chat' | 'documents';
}

/**
 * Unified top-level navigation switcher between Chat and Documents.
 * Provides identical visual appearance, responsive styling, and clear active indication across both pages.
 */
export function MainNav({ className, activeSection }: MainNavProps) {
  const pathname = usePathname();

  const isChatActive = activeSection ? activeSection === 'chat' : pathname === '/';
  const isDocumentsActive = activeSection
    ? activeSection === 'documents'
    : pathname.startsWith('/documents');

  return (
    <nav
      aria-label="Main Navigation"
      className={cn(
        'inline-flex items-center rounded-xl bg-zinc-100/90 p-1 dark:bg-zinc-900/90 border border-zinc-200/80 dark:border-zinc-800 shadow-2xs backdrop-blur-xs',
        className
      )}
    >
      {/* Chat Link */}
      <Link
        href="/"
        aria-current={isChatActive ? 'page' : undefined}
        className={cn(
          'flex items-center gap-1.5 rounded-lg px-2.5 sm:px-3 py-1.5 text-xs transition-all duration-150 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500',
          isChatActive
            ? 'bg-white font-semibold text-zinc-900 shadow-xs ring-1 ring-zinc-200/60 dark:bg-zinc-800 dark:text-zinc-50 dark:ring-zinc-700/60'
            : 'font-medium text-zinc-600 hover:text-zinc-900 hover:bg-zinc-200/50 dark:text-zinc-400 dark:hover:text-zinc-100 dark:hover:bg-zinc-800/50'
        )}
      >
        <svg
          className={cn(
            'h-3.5 w-3.5 shrink-0 transition-colors',
            isChatActive
              ? 'text-blue-600 dark:text-blue-400'
              : 'text-zinc-400 dark:text-zinc-500'
          )}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
        </svg>
        <span>Chat</span>
      </Link>

      {/* Documents Link */}
      <Link
        href="/documents"
        aria-current={isDocumentsActive ? 'page' : undefined}
        className={cn(
          'flex items-center gap-1.5 rounded-lg px-2.5 sm:px-3 py-1.5 text-xs transition-all duration-150 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500',
          isDocumentsActive
            ? 'bg-white font-semibold text-zinc-900 shadow-xs ring-1 ring-zinc-200/60 dark:bg-zinc-800 dark:text-zinc-50 dark:ring-zinc-700/60'
            : 'font-medium text-zinc-600 hover:text-zinc-900 hover:bg-zinc-200/50 dark:text-zinc-400 dark:hover:text-zinc-100 dark:hover:bg-zinc-800/50'
        )}
      >
        <svg
          className={cn(
            'h-3.5 w-3.5 shrink-0 transition-colors',
            isDocumentsActive
              ? 'text-violet-600 dark:text-violet-400'
              : 'text-zinc-400 dark:text-zinc-500'
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
        <span>Documents</span>
      </Link>
    </nav>
  );
}
