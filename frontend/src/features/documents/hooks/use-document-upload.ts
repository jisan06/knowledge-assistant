'use client';

import { useState, useCallback } from 'react';
import {
  DocumentUploadResponse,
  DocumentApiError,
  UploadStatus,
  ValidationError,
  MAX_UPLOAD_BYTES,
} from '../types/document';
import { documentApi } from '../services/document-api';

export interface UseDocumentUploadReturn {
  status: UploadStatus;
  selectedFile: File | null;
  validationError: ValidationError;
  uploadError: string | null;
  result: DocumentUploadResponse | null;
  selectFile: (file: File | null) => void;
  upload: () => Promise<void>;
  reset: () => void;
}

/**
 * Hook that manages the full document upload lifecycle:
 *   idle → validating → uploading → success | error
 */
export function useDocumentUpload(): UseDocumentUploadReturn {
  const [status, setStatus] = useState<UploadStatus>('idle');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [validationError, setValidationError] = useState<ValidationError>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [result, setResult] = useState<DocumentUploadResponse | null>(null);

  /** Validate and stage a file for upload. */
  const selectFile = useCallback((file: File | null) => {
    // Reset previous results whenever a new file is selected
    setResult(null);
    setUploadError(null);
    setValidationError(null);

    if (!file) {
      setSelectedFile(null);
      setStatus('idle');
      return;
    }

    const isPdf =
      file.name.toLowerCase().endsWith('.pdf') || file.type === 'application/pdf';

    if (!isPdf) {
      setValidationError('not_pdf');
      setSelectedFile(null);
      setStatus('idle');
      return;
    }

    if (file.size === 0) {
      setValidationError('empty_file');
      setSelectedFile(null);
      setStatus('idle');
      return;
    }

    if (file.size > MAX_UPLOAD_BYTES) {
      setValidationError('too_large');
      setSelectedFile(null);
      setStatus('idle');
      return;
    }

    setSelectedFile(file);
    setStatus('idle');
  }, []);

  /** Send the staged file to the backend extraction endpoint. */
  const upload = useCallback(async () => {
    if (!selectedFile || status === 'uploading') return;

    setStatus('uploading');
    setUploadError(null);
    setResult(null);

    try {
      const data = await documentApi.uploadPdf(selectedFile);
      setResult(data);
      setStatus('success');
    } catch (err: unknown) {
      const apiError = err as DocumentApiError;
      setUploadError(apiError.message || 'An unexpected error occurred. Please try again.');
      setStatus('error');
    }
  }, [selectedFile, status]);

  /** Reset everything back to the initial idle state. */
  const reset = useCallback(() => {
    setStatus('idle');
    setSelectedFile(null);
    setValidationError(null);
    setUploadError(null);
    setResult(null);
  }, []);

  return {
    status,
    selectedFile,
    validationError,
    uploadError,
    result,
    selectFile,
    upload,
    reset,
  };
}
