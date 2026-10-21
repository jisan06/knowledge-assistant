'use client';

import React, { useState } from 'react';
import { DocumentPage } from '../types/document';

interface PageResultProps {
  page: DocumentPage;
  isFirst: boolean;
}

/**
 * Renders extracted text for a single PDF page.
 * Collapsed by default if the page is long; always expanded for the first page.
 */
export function PageResult({ page, isFirst }: PageResultProps) {
  const isLong = page.text.length > 600;
  const [expanded, setExpanded] = useState(!isLong || isFirst);
  const displayText = expanded || !isLong ? page.text : `${page.text.slice(0, 600)}…`;

  const hasText = page.char_count > 0;

  return (
    <div className="overflow-hidden rounded-xl border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900 animate-fadeIn">
      {/* Page header */}
      <div className="flex items-center justify-between border-b border-zinc-100 bg-zinc-50/60 px-4 py-2.5 dark:border-zinc-800 dark:bg-zinc-800/40">
        <div className="flex items-center gap-2">
          <span className="inline-flex h-6 w-6 items-center justify-center rounded-md bg-blue-100 text-[11px] font-semibold text-blue-700 dark:bg-blue-950/60 dark:text-blue-300">
            {page.page_number}
          </span>
          <span className="text-xs font-medium text-zinc-600 dark:text-zinc-400">
            Page {page.page_number}
          </span>
        </div>
        <span className="text-[11px] text-zinc-400 dark:text-zinc-500">
          {hasText
            ? `${page.char_count.toLocaleString()} chars${
                page.chunk_count !== undefined
                  ? ` · ${page.chunk_count} ${page.chunk_count === 1 ? 'chunk' : 'chunks'}`
                  : ''
              }`
            : 'No text'}
        </span>
      </div>

      {/* Page body */}
      <div className="px-4 py-3">
        {hasText ? (
          <>
            <pre className="whitespace-pre-wrap break-words font-mono text-xs leading-relaxed text-zinc-700 dark:text-zinc-300">
              {displayText}
            </pre>
            {isLong && (
              <button
                type="button"
                onClick={() => setExpanded((prev) => !prev)}
                className="mt-2 text-[11px] font-medium text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 focus:outline-none focus:underline"
              >
                {expanded ? '↑ Show less' : `↓ Show all ${page.char_count.toLocaleString()} characters`}
              </button>
            )}
          </>
        ) : (
          <p className="py-2 text-xs italic text-zinc-400 dark:text-zinc-500">
            No extractable text on this page. It may contain only images or scanned content.
          </p>
        )}
      </div>
    </div>
  );
}
