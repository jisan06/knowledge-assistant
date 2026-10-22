'use client';

import React from 'react';
import { MainNav } from '@/components/main-nav';
import { useDocumentUpload } from '../hooks/use-document-upload';
import { UploadZone } from './upload-zone';
import { ExtractionResults } from './extraction-results';
import { DocumentList } from './document-list';

/**
 * Top-level page component for the Documents / PDF extraction feature.
 * Orchestrates the upload hook and renders the appropriate view for each state.
 */
export function DocumentUploadPage() {
  const { status, selectedFile, validationError, uploadError, result, selectFile, upload, reset } =
    useDocumentUpload();

  const [refreshTrigger, setRefreshTrigger] = React.useState(0);

  React.useEffect(() => {
    if (status === 'success') {
      setRefreshTrigger((prev) => prev + 1);
    }
  }, [status]);

  const isUploading = status === 'uploading';

  return (
    <div className="flex min-h-screen flex-col bg-zinc-50 font-sans text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100">
      {/* Page header — mirrors the chat header height and styling */}
      <header className="sticky top-0 z-10 flex items-center justify-between border-b border-zinc-200/80 bg-white/80 px-4 py-3 backdrop-blur-md dark:border-zinc-800 dark:bg-zinc-950/80 sm:px-6">
        <div className="flex items-center gap-3">
          {/* Documents icon */}
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-tr from-violet-600 via-indigo-600 to-blue-500 text-white shadow-sm">
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z" />
            </svg>
          </div>

          <div className="min-w-0">
            <h1 className="text-sm font-semibold tracking-tight text-zinc-900 dark:text-zinc-50 sm:text-base">
              Documents
            </h1>
            <p className="text-[11px] text-zinc-500 dark:text-zinc-400">PDF Text Extraction & Knowledge Management</p>
          </div>
        </div>

        {/* Navigation Switcher */}
        <div className="flex items-center gap-2">
          <MainNav activeSection="documents" />
        </div>
      </header>

      {/* Main content */}
      <main className="mx-auto w-full max-w-3xl flex-1 px-4 py-8 sm:px-6">
        {/* Upload zone — always visible until we have a result */}
        {status !== 'success' && (
          <section className="space-y-6">
            <div>
              <h2 className="text-lg font-semibold text-zinc-800 dark:text-zinc-100">
                Upload a PDF
              </h2>
              <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
                Select or drag and drop a PDF document. The text will be extracted page by page and indexed into vector search.
              </p>
            </div>

            <UploadZone
              selectedFile={selectedFile}
              validationError={validationError}
              isUploading={isUploading}
              onFileSelect={selectFile}
              onUpload={upload}
              onReset={reset}
            />

            {/* Upload error banner */}
            {status === 'error' && uploadError && (
              <div
                role="alert"
                className="flex items-start gap-3 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800 dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-300"
              >
                <svg
                  className="mt-px h-4 w-4 shrink-0 text-red-500"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                  aria-hidden="true"
                >
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
                <div className="min-w-0">
                  <p className="font-medium">Upload failed</p>
                  <p className="mt-0.5 text-xs">{uploadError}</p>
                </div>
              </div>
            )}
          </section>
        )}

        {/* Extraction results */}
        {status === 'success' && result && (
          <ExtractionResults result={result} onReset={reset} />
        )}

        {/* Document management list */}
        <DocumentList refreshTrigger={refreshTrigger} />
      </main>
    </div>
  );
}
