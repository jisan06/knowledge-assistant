'use client';

import React, { useRef, useId } from 'react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { ValidationError, MAX_UPLOAD_BYTES } from '../types/document';

interface UploadZoneProps {
  selectedFile: File | null;
  validationError: ValidationError;
  isUploading: boolean;
  onFileSelect: (file: File | null) => void;
  onUpload: () => void;
  onReset: () => void;
}

/** Formats bytes into a readable MB/KB string. */
function formatSize(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${bytes} B`;
}

const MAX_MB = MAX_UPLOAD_BYTES / (1024 * 1024);

/**
 * Drop zone and file picker for PDF upload.
 * Handles drag-and-drop, file input selection, and client-side validation feedback.
 */
export function UploadZone({
  selectedFile,
  validationError,
  isUploading,
  onFileSelect,
  onUpload,
  onReset,
}: UploadZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const inputId = useId();
  const [isDragging, setIsDragging] = React.useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onFileSelect(e.target.files?.[0] ?? null);
    // Reset input value so the same file can be re-selected after reset
    if (inputRef.current) inputRef.current.value = '';
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    onFileSelect(e.dataTransfer.files?.[0] ?? null);
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => setIsDragging(false);

  const openFilePicker = () => inputRef.current?.click();

  const validationMessages: Record<NonNullable<ValidationError>, string> = {
    not_pdf: 'Only PDF files are supported. Please select a .pdf file.',
    too_large: `File is too large. Maximum allowed size is ${MAX_MB} MB.`,
    empty_file: 'The selected file is empty (0 bytes). Please select a valid PDF.',
  };

  return (
    <div className="space-y-4">
      {/* Drop zone */}
      <div
        role="button"
        tabIndex={0}
        aria-label="Drop PDF here or click to select"
        onClick={openFilePicker}
        onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && openFilePicker()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={cn(
          'group flex cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed px-6 py-12 text-center transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2',
          isDragging
            ? 'border-blue-500 bg-blue-50 dark:bg-blue-950/30'
            : validationError
            ? 'border-red-300 bg-red-50/50 dark:border-red-800 dark:bg-red-950/20'
            : 'border-zinc-200 bg-zinc-50/50 hover:border-blue-400 hover:bg-blue-50/30 dark:border-zinc-700 dark:bg-zinc-900/30 dark:hover:border-blue-600 dark:hover:bg-blue-950/20'
        )}
      >
        {/* Icon */}
        <div
          className={cn(
            'pointer-events-none flex h-14 w-14 items-center justify-center rounded-2xl transition-colors',
            isDragging
              ? 'bg-blue-100 text-blue-600 dark:bg-blue-900/50 dark:text-blue-400'
              : validationError
              ? 'bg-red-100 text-red-500 dark:bg-red-900/30 dark:text-red-400'
              : 'bg-zinc-100 text-zinc-400 group-hover:bg-blue-100 group-hover:text-blue-500 dark:bg-zinc-800 dark:group-hover:bg-blue-900/40 dark:group-hover:text-blue-400'
          )}
        >
          <svg className="h-7 w-7" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m6.75 12-3-3m0 0-3 3m3-3v6m-1.5-15H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z"
            />
          </svg>
        </div>

        {/* Text */}
        <div className="pointer-events-none">
          <p className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
            {isDragging ? 'Release to upload' : 'Drop your PDF here'}
          </p>
          <p className="mt-1 text-xs text-zinc-400 dark:text-zinc-500">
            or{' '}
            <span className="font-medium text-blue-600 dark:text-blue-400">browse to choose a file</span>
          </p>
          <p className="mt-1.5 text-[11px] text-zinc-400 dark:text-zinc-500">
            PDF only · max {MAX_MB} MB
          </p>
        </div>

        <input
          ref={inputRef}
          id={inputId}
          type="file"
          accept=".pdf,application/pdf"
          className="sr-only"
          onChange={handleFileChange}
          aria-label="Select PDF file"
        />
      </div>

      {/* Validation error */}
      {validationError && (
        <div
          role="alert"
          className="flex items-start gap-2 rounded-xl border border-red-200 bg-red-50 p-3 text-xs text-red-700 dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-300"
        >
          <svg className="mt-px h-3.5 w-3.5 shrink-0" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
          </svg>
          {validationMessages[validationError]}
        </div>
      )}

      {/* Selected file info + action buttons */}
      {selectedFile && (
        <div className="flex items-center justify-between gap-3 rounded-xl border border-zinc-200 bg-white px-4 py-3 dark:border-zinc-800 dark:bg-zinc-900">
          <div className="flex min-w-0 items-center gap-3">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-blue-50 text-blue-600 dark:bg-blue-950/40 dark:text-blue-400">
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5} aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z" />
              </svg>
            </div>
            <div className="min-w-0">
              <p className="truncate text-sm font-medium text-zinc-800 dark:text-zinc-100" title={selectedFile.name}>
                {selectedFile.name}
              </p>
              <p className="text-xs text-zinc-400 dark:text-zinc-500">{formatSize(selectedFile.size)}</p>
            </div>
          </div>

          <div className="flex shrink-0 items-center gap-2">
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); onReset(); }}
              disabled={isUploading}
              aria-label="Remove selected file"
              className="rounded-lg p-1.5 text-zinc-400 hover:bg-zinc-100 hover:text-zinc-600 disabled:opacity-40 dark:hover:bg-zinc-800 dark:hover:text-zinc-300 focus:outline-none focus:ring-2 focus:ring-zinc-400"
            >
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
            <Button
              id="upload-pdf-button"
              onClick={(e) => { e.stopPropagation(); onUpload(); }}
              isLoading={isUploading}
              disabled={isUploading}
              size="sm"
            >
              {isUploading ? 'Extracting…' : 'Extract Text'}
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
