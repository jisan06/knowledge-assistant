import { DocumentUploadPage } from '@/features/documents';

export const metadata = {
  title: 'Documents — AI Knowledge Assistant',
  description: 'Upload a PDF to extract and view its text content page by page.',
};

export default function DocumentsPage() {
  return <DocumentUploadPage />;
}
