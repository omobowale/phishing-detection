import { apiFetch } from './client';
import type { Metrics } from './types';

export function getMetrics(): Promise<Metrics> {
  return apiFetch<Metrics>('/metrics');
}
