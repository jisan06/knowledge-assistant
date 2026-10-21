/**
 * Document Feature Type Definitions
 */

/** One extracted page from the PDF, 1-based page numbering. */
export interface DocumentPage {
  page_number: number;
  text: string;
  char_count: number;
}

/** Successful response from POST /api/documents/upload */
export interface DocumentUploadResponse {
  filename: string;
  total_pages: number;
  total_characters: number;
  has_extractable_text: boolean;
  pages: DocumentPage[];
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
