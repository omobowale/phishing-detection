import { apiFetch } from './client';
import type { DetectRequest, DetectResponse } from './types';

export function detect(payload: DetectRequest): Promise<DetectResponse> {
  return apiFetch<DetectResponse>('/detect', { method: 'POST', body: payload });
}
