import { DocumentApiError, DocumentUploadResponse } from '../types/document';

/**
 * Service for all HTTP interactions with the document extraction backend.
 * Uses the same base URL strategy as the existing ChatApiService.
 */
class DocumentApiService {
  private get cleanBaseUrl(): string {
    const base = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000';
    return base.replace(/\/+$/, '');
  }

  /**
   * Upload a single PDF file for text extraction.
   *
   * Sends a multipart/form-data POST to /api/documents/upload.
   * The Content-Type header and multipart boundary are generated
   * automatically by fetch — we must NOT set it manually.
   */
  async uploadPdf(file: File, index: boolean = true): Promise<DocumentUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    let response: Response;
    try {
      response = await fetch(`${this.cleanBaseUrl}/api/documents/upload?index=${index}`, {
        method: 'POST',
        // No Content-Type header — fetch generates the correct multipart boundary
        body: formData,
      });
    } catch {
      throw {
        message: 'Unable to reach the server. Please verify the backend is running.',
      } as DocumentApiError;
    }

    if (!response.ok) {
      // Try to parse structured error detail from FastAPI
      let detail = '';
      try {
        const errorBody = await response.json();
        if (typeof errorBody?.detail === 'string') {
          detail = errorBody.detail;
        } else if (Array.isArray(errorBody?.detail) && errorBody.detail.length > 0) {
          detail = errorBody.detail
            .map((item: { msg?: string }) => item.msg || 'Validation error')
            .join('; ');
        }
      } catch {
        // Response body could not be parsed as JSON
      }

      // Fallback to clear, status-code-based messaging if no server detail was provided
      if (!detail) {
        if (response.status === 400) {
          detail = 'Invalid request. Only PDF files are accepted.';
        } else if (response.status === 413) {
          detail = 'File is too large. Maximum allowed size is 10 MB.';
        } else if (response.status === 422) {
          detail = 'The file could not be parsed. It may be corrupted or malformed.';
        } else if (response.status >= 500) {
          detail = 'The server encountered an error. Please try again later.';
        } else {
          detail = 'Upload failed. Please try again.';
        }
      }

      throw { message: detail, statusCode: response.status } as DocumentApiError;
    }

    try {
      const data: DocumentUploadResponse = await response.json();
      return data;
    } catch {
      throw {
        message: 'Received an unexpected response format from the server.',
      } as DocumentApiError;
    }
  }
}

export const documentApi = new DocumentApiService();
