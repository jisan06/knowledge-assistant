/**
 * Utility functions for frontend application
 */

type ClassValue = string | number | boolean | undefined | null | { [key: string]: boolean | undefined | null } | ClassValue[];

/**
 * Combines and normalizes CSS class names cleanly without requiring external libraries.
 */
export function cn(...inputs: ClassValue[]): string {
  const classes: string[] = [];

  for (const input of inputs) {
    if (!input) continue;

    if (typeof input === 'string' || typeof input === 'number') {
      classes.push(String(input));
    } else if (Array.isArray(input)) {
      const inner = cn(...input);
      if (inner) classes.push(inner);
    } else if (typeof input === 'object') {
      for (const [key, value] of Object.entries(input)) {
        if (value) classes.push(key);
      }
    }
  }

  return classes.join(' ').replace(/\s+/g, ' ').trim();
}

/**
 * Format timestamp to a user-friendly local time string (e.g., "10:30 AM").
 */
export function formatTime(date: Date | string | number): string {
  const d = typeof date === 'object' ? date : new Date(date);
  if (isNaN(d.getTime())) return '';

  return new Intl.DateTimeFormat('en-US', {
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
  }).format(d);
}

/**
 * Generates a unique identifier for messages.
 */
export function generateId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `msg_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
}
