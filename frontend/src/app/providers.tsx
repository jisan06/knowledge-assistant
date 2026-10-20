'use client';

import React from 'react';

interface ProvidersProps {
  children: React.ReactNode;
}

/**
 * Global client providers wrapper.
 * Centralizes context providers (theme, queries, toasts) without bloating layout.tsx.
 */
export function Providers({ children }: ProvidersProps) {
  return <>{children}</>;
}
