import { apiFetch } from './client';
import type { User } from './types';

interface TokenResponse {
  access_token: string;
  token_type: string;
}

export function register(name: string, email: string, password: string): Promise<User> {
  return apiFetch<User>('/auth/register', {
    method: 'POST',
    body: { name, email, password },
    auth: false,
  });
}

export async function login(email: string, password: string): Promise<string> {
  const form = new URLSearchParams();
  form.set('username', email);
  form.set('password', password);
  const { access_token } = await apiFetch<TokenResponse>('/auth/login', {
    method: 'POST',
    body: form,
    auth: false,
    isForm: true,
  });
  return access_token;
}

export function me(): Promise<User> {
  return apiFetch<User>('/auth/me');
}
