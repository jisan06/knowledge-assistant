/**
 * Document Feature Type Definitions
 */

/** One extracted text chunk from a page. */
export interface DocumentChunk {
  chunk_id: string;
  filename: string;
  page_number: number;
  chunk_index: number;
  text: string;
  char_count: number;
  start_char: number;
  end_char: number;
  embedding?: number[];
}

/** One extracted page from the PDF, 1-based page numbering. */
export interface DocumentPage {
  page_number: number;
  text: string;
  char_count: number;
  chunk_count?: number;
  chunks?: DocumentChunk[];
}

/** Successful response from POST /api/documents/upload */
export interface DocumentUploadResponse {
  filename: string;
  total_pages: number;
  total_characters: number;
  has_extractable_text: boolean;
  pages: DocumentPage[];
  total_chunks?: number;
  chunks?: DocumentChunk[];
  indexed_chunks?: number;
  message: string;
}

/** Domain-level error shape used inside the feature */
export interface DocumentApiError {
  message: string;
  statusCode?: number;
}

/** Client-side upload state machine */
export type UploadStatus = 'idle' | 'validating' | 'uploading' | 'success' | 'error';

/** Client-side validation error types */
export type ValidationError =
  | 'not_pdf'
  | 'too_large'
  | 'empty_file'
  | null;

export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024; // 10 MB — mirrors backend setting
