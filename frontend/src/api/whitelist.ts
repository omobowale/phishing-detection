import { apiFetch } from './client';
import type { WhitelistEntry } from './types';

export function listWhitelist(): Promise<WhitelistEntry[]> {
  return apiFetch<WhitelistEntry[]>('/whitelist');
}

export function addWhitelistEntry(domain: string): Promise<WhitelistEntry> {
  return apiFetch<WhitelistEntry>('/whitelist', { method: 'POST', body: { domain } });
}

export function deleteWhitelistEntry(id: number): Promise<void> {
  return apiFetch<void>(`/whitelist/${id}`, { method: 'DELETE' });
}
