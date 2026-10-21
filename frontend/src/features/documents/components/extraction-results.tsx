'use client';

import React from 'react';
import { DocumentUploadResponse } from '../types/document';
import { PageResult } from './page-result';
import { Button } from '@/components/ui/button';

interface ExtractionResultsProps {
  result: DocumentUploadResponse;
  onReset: () => void;
}

/**
 * Displays the full extraction report: summary banner followed by per-page results.
 */
export function ExtractionResults({ result, onReset }: ExtractionResultsProps) {
  const { filename, total_pages, total_characters, has_extractable_text, message, pages } = result;

  return (
    <div className="space-y-5 animate-fadeIn">
      {/* Summary card */}
      <div
        className={`rounded-2xl border p-4 ${
          has_extractable_text
            ? 'border-emerald-200 bg-emerald-50 dark:border-emerald-900/60 dark:bg-emerald-950/30'
            : 'border-amber-200 bg-amber-50 dark:border-amber-900/60 dark:bg-amber-950/30'
        }`}
      >
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="flex items-start gap-3">
            {/* Status icon */}
            <div
              className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-xl ${
                has_extractable_text
                  ? 'bg-emerald-100 text-emerald-600 dark:bg-emerald-900/40 dark:text-emerald-400'
                  : 'bg-amber-100 text-amber-600 dark:bg-amber-900/40 dark:text-amber-400'
              }`}
            >
              {has_extractable_text ? (
                <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              ) : (
                <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
                </svg>
              )}
            </div>

            {/* File details */}
            <div className="min-w-0">
              <p className="truncate text-sm font-semibold text-zinc-800 dark:text-zinc-100" title={filename}>
                {filename}
              </p>
              <p
                className={`mt-0.5 text-xs ${
                  has_extractable_text
                    ? 'text-emerald-700 dark:text-emerald-400'
                    : 'text-amber-700 dark:text-amber-400'
                }`}
              >
                {message}
              </p>
            </div>
          </div>

          {/* Stats chips */}
          <div className="flex flex-wrap gap-2">
            <StatChip label="Pages" value={String(total_pages)} />
            {has_extractable_text && (
              <StatChip label="Characters" value={total_characters.toLocaleString()} />
            )}
          </div>
        </div>
      </div>

      {/* Per-page results */}
      <div className="space-y-3">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-400 dark:text-zinc-500">
          Extracted Pages
        </h2>
        {pages.map((page, idx) => (
          <PageResult key={page.page_number} page={page} isFirst={idx === 0} />
        ))}
      </div>

      {/* Action to upload another document */}
      <div className="pt-2">
        <Button variant="outline" size="sm" onClick={onReset} id="upload-another-button">
          <svg className="mr-1.5 h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 8.25H7.5a2.25 2.25 0 0 0-2.25 2.25v9a2.25 2.25 0 0 0 2.25 2.25h9a2.25 2.25 0 0 0 2.25-2.25v-9a2.25 2.25 0 0 0-2.25-2.25H15m0-3-3-3m0 0-3 3m3-3V15" />
          </svg>
          Upload Another PDF
        </Button>
      </div>
    </div>
  );
}

function StatChip({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-zinc-200/70 bg-white/70 px-2.5 py-1 dark:border-zinc-700/50 dark:bg-zinc-800/60">
      <span className="text-[10px] font-medium uppercase tracking-wide text-zinc-400 dark:text-zinc-500">
        {label}
      </span>
      <p className="text-sm font-semibold text-zinc-700 dark:text-zinc-200">{value}</p>
    </div>
  );
}
