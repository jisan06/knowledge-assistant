'use client';

import React, { useEffect, useState, useCallback } from 'react';
import { DocumentItem } from '../types/document';
import { documentApi } from '../services/document-api';
import { Button } from '@/components/ui/button';

interface DocumentListProps {
  refreshTrigger?: number;
  onDocumentDeleted?: (deletedId: string) => void;
}

/**
 * Lists all uploaded documents with metadata and inline deletion with confirmation.
 */
export function DocumentList({ refreshTrigger, onDocumentDeleted }: DocumentListProps) {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const loadDocuments = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await documentApi.getDocuments();
      setDocuments(data);
    } catch (err: unknown) {
      const apiErr = err as { message?: string };
      setError(apiErr?.message || 'Failed to load documents.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadDocuments();
  }, [loadDocuments, refreshTrigger]);

  const handleDelete = async (doc: DocumentItem) => {
    setDeletingId(doc.id);
    setDeleteError(null);
    try {
      await documentApi.deleteDocument(doc.id);
      setDocuments((prev) => prev.filter((d) => d.id !== doc.id));
      setConfirmDeleteId(null);
      if (onDocumentDeleted) {
        onDocumentDeleted(doc.id);
      }
    } catch (err: unknown) {
      const apiErr = err as { message?: string };
      setDeleteError(apiErr?.message || `Failed to delete '${doc.filename}'.`);
    } finally {
      setDeletingId(null);
    }
  };

  const formatDate = (dateString: string) => {
    try {
      const d = new Date(dateString);
      return d.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return dateString;
    }
  };

  return (
    <section className="mt-10 space-y-4">
      {/* Section Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <h2 className="text-base font-semibold text-zinc-900 dark:text-zinc-100">
            Knowledge Base Documents
          </h2>
          {!isLoading && (
            <span className="inline-flex items-center rounded-full bg-zinc-100 px-2 py-0.5 text-xs font-medium text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300">
              {documents.length}
            </span>
          )}
        </div>

        <Button
          variant="ghost"
          size="sm"
          onClick={loadDocuments}
          disabled={isLoading}
          className="text-xs text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100"
          id="refresh-documents-btn"
        >
          <svg
            className={`mr-1.5 h-3.5 w-3.5 ${isLoading ? 'animate-spin' : ''}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0 3.181 3.183a8.25 8.25 0 0 0 13.803-3.7M4.031 9.865a8.25 8.25 0 0 1 13.803-3.7l3.181 3.182m0-4.991v4.99"
            />
          </svg>
          Refresh
        </Button>
      </div>

      {/* Delete error notification */}
      {deleteError && (
        <div
          role="alert"
          className="flex items-center justify-between rounded-xl border border-red-200 bg-red-50 px-3.5 py-2.5 text-xs text-red-800 dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-300"
        >
          <span>{deleteError}</span>
          <button
            onClick={() => setDeleteError(null)}
            className="ml-2 font-semibold hover:underline"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Loading state */}
      {isLoading && (
        <div className="flex flex-col items-center justify-center rounded-2xl border border-dashed border-zinc-200 p-8 text-center dark:border-zinc-800">
          <div className="h-6 w-6 animate-spin rounded-full border-2 border-indigo-600 border-t-transparent dark:border-indigo-400" />
          <p className="mt-3 text-xs text-zinc-500 dark:text-zinc-400">Loading indexed documents...</p>
        </div>
      )}

      {/* Error state */}
      {!isLoading && error && (
        <div className="rounded-2xl border border-amber-200 bg-amber-50 p-6 text-center dark:border-amber-900/60 dark:bg-amber-950/30">
          <p className="text-sm font-medium text-amber-800 dark:text-amber-200">{error}</p>
          <Button
            variant="outline"
            size="sm"
            onClick={loadDocuments}
            className="mt-3 text-xs"
          >
            Try Again
          </Button>
        </div>
      )}

      {/* Empty state */}
      {!isLoading && !error && documents.length === 0 && (
        <div className="flex flex-col items-center justify-center rounded-2xl border border-dashed border-zinc-200 bg-white/50 p-8 text-center dark:border-zinc-800 dark:bg-zinc-900/30">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-zinc-100 text-zinc-400 dark:bg-zinc-800 dark:text-zinc-500">
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z" />
            </svg>
          </div>
          <p className="mt-3 text-sm font-medium text-zinc-700 dark:text-zinc-300">
            No indexed documents yet
          </p>
          <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
            Upload a PDF above to enable document-grounded AI search and retrieval.
          </p>
        </div>
      )}

      {/* Populated Document List */}
      {!isLoading && !error && documents.length > 0 && (
        <div className="divide-y divide-zinc-200/80 rounded-2xl border border-zinc-200/80 bg-white shadow-xs dark:divide-zinc-800 dark:border-zinc-800 dark:bg-zinc-900/60">
          {documents.map((doc) => {
            const isConfirming = confirmDeleteId === doc.id;
            const isDeleting = deletingId === doc.id;

            return (
              <div
                key={doc.id}
                className="flex flex-col gap-3 p-4 transition-colors hover:bg-zinc-50/50 dark:hover:bg-zinc-800/30 sm:flex-row sm:items-center sm:justify-between"
              >
                {/* File info and badges */}
                <div className="flex items-start gap-3 min-w-0">
                  <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-violet-100/70 text-violet-600 dark:bg-violet-950/60 dark:text-violet-400">
                    <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z" />
                    </svg>
                  </div>

                  <div className="min-w-0 flex-1">
                    <p
                      className="truncate text-sm font-medium text-zinc-900 dark:text-zinc-100"
                      title={doc.filename}
                    >
                      {doc.filename}
                    </p>

                    <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px] text-zinc-500 dark:text-zinc-400">
                      <span>{doc.total_pages} page{doc.total_pages === 1 ? '' : 's'}</span>
                      <span>•</span>
                      <span>{doc.indexed_chunks} indexed chunk{doc.indexed_chunks === 1 ? '' : 's'}</span>
                      <span>•</span>
                      <span>{doc.total_characters.toLocaleString()} chars</span>
                      <span>•</span>
                      <span>{formatDate(doc.created_at)}</span>
                    </div>
                  </div>
                </div>

                {/* Right side status badge and actions */}
                <div className="flex items-center gap-2 shrink-0 self-end sm:self-center">
                  {doc.indexed_chunks > 0 ? (
                    <span className="inline-flex items-center gap-1 rounded-md bg-emerald-50 px-2 py-1 text-[11px] font-medium text-emerald-700 ring-1 ring-inset ring-emerald-600/20 dark:bg-emerald-950/40 dark:text-emerald-400">
                      <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
                      Indexed
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 rounded-md bg-zinc-100 px-2 py-1 text-[11px] font-medium text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400">
                      Not Indexed
                    </span>
                  )}

                  {/* Delete with inline confirmation */}
                  {isConfirming ? (
                    <div className="flex items-center gap-1.5 bg-red-50 dark:bg-red-950/40 p-1 rounded-lg border border-red-200 dark:border-red-900/60">
                      <span className="text-[11px] font-medium text-red-700 dark:text-red-400 px-1">
                        Confirm delete?
                      </span>
                      <Button
                        size="sm"
                        variant="destructive"
                        disabled={isDeleting}
                        onClick={() => handleDelete(doc)}
                        className="h-7 px-2 text-xs"
                        id={`confirm-delete-${doc.id}`}
                      >
                        {isDeleting ? 'Deleting...' : 'Delete'}
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        disabled={isDeleting}
                        onClick={() => setConfirmDeleteId(null)}
                        className="h-7 px-2 text-xs"
                      >
                        Cancel
                      </Button>
                    </div>
                  ) : (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setConfirmDeleteId(doc.id)}
                      className="h-8 w-8 p-0 text-zinc-400 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-950/40 dark:hover:text-red-400"
                      title="Delete document"
                      id={`delete-doc-${doc.id}`}
                    >
                      <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 0 0-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 0 1 3.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 0 0-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 0 0-7.5 0" />
                      </svg>
                      <span className="sr-only">Delete {doc.filename}</span>
                    </Button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}
