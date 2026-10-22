'use client';

import React from 'react';
import Link from 'next/link';
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
  const {
    filename,
    total_pages,
    total_characters,
    has_extractable_text,
    message,
    pages,
    total_chunks,
    indexed_chunks,
  } = result;

  const isIndexed = typeof indexed_chunks === 'number' && indexed_chunks > 0;

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
              <>
                <StatChip label="Characters" value={total_characters.toLocaleString()} />
                {total_chunks !== undefined && (
                  <StatChip label="Chunks" value={String(total_chunks)} />
                )}
                {isIndexed && (
                  <StatChip label="Indexed" value={`${indexed_chunks} chunks`} variant="success" />
                )}
              </>
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

      {/* Action buttons */}
      <div className="flex flex-wrap items-center gap-3 pt-2">
        <Button variant="outline" size="sm" onClick={onReset} id="upload-another-button">
          <svg className="mr-1.5 h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 8.25H7.5a2.25 2.25 0 0 0-2.25 2.25v9a2.25 2.25 0 0 0 2.25 2.25h9a2.25 2.25 0 0 0 2.25-2.25v-9a2.25 2.25 0 0 0-2.25-2.25H15m0-3-3-3m0 0-3 3m3-3V15" />
          </svg>
          Upload Another PDF
        </Button>

        {isIndexed && (
          <Link href="/">
            <Button
              size="sm"
              className="bg-gradient-to-r from-violet-600 to-indigo-600 text-white shadow-sm hover:from-violet-500 hover:to-indigo-500"
              id="chat-with-document-button"
            >
              <svg className="mr-1.5 h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.129.166 2.27.293 3.423.379.35.026.67.21.865.501L12 21l2.755-4.133a1.14 1.14 0 0 1 .865-.501 48.172 48.172 0 0 0 3.423-.379c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0 0 12 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v5.769Z" />
              </svg>
              Ask Questions in Chat
            </Button>
          </Link>
        )}
      </div>
    </div>
  );
}

function StatChip({
  label,
  value,
  variant = 'default',
}: {
  label: string;
  value: string;
  variant?: 'default' | 'success';
}) {
  const isSuccess = variant === 'success';
  return (
    <div
      className={`rounded-lg border px-2.5 py-1 ${
        isSuccess
          ? 'border-emerald-300 bg-emerald-100/70 text-emerald-800 dark:border-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300'
          : 'border-zinc-200/70 bg-white/70 text-zinc-700 dark:border-zinc-700/50 dark:bg-zinc-800/60 dark:text-zinc-200'
      }`}
    >
      <span
        className={`text-[10px] font-medium uppercase tracking-wide ${
          isSuccess ? 'text-emerald-700 dark:text-emerald-400' : 'text-zinc-400 dark:text-zinc-500'
        }`}
      >
        {label}
      </span>
      <p className="text-sm font-semibold">{value}</p>
    </div>
  );
}
