import { apiFetch } from './client';
import type { DetectionLogEntry, LogFilters, PaginatedLogs } from './types';

export function listLogs(filters: LogFilters): Promise<PaginatedLogs> {
  const params = new URLSearchParams();
  Object.entries(filters).forEach(([key, value]) => {
    if (value !== undefined && value !== '') params.set(key, String(value));
  });
  const query = params.toString();
  return apiFetch<PaginatedLogs>(`/logs${query ? `?${query}` : ''}`);
}

export function getLog(id: number): Promise<DetectionLogEntry> {
  return apiFetch<DetectionLogEntry>(`/logs/${id}`);
}
