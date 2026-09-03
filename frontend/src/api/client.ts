const API_BASE = '/api/v1';
const TOKEN_KEY = 'phishing_detection_token';

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

function extractDetailMessage(detail: unknown): string | null {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    const messages = detail
      .map((entry) => (entry && typeof entry === 'object' && 'msg' in entry ? String((entry as { msg: unknown }).msg) : null))
      .filter((msg): msg is string => Boolean(msg));
    if (messages.length) return messages.join('; ');
  }
  return null;
}

interface RequestOptions {
  method?: string;
  body?: unknown;
  auth?: boolean;
  isForm?: boolean;
}

export async function apiFetch<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = 'GET', body, auth = true, isForm = false } = options;
  const headers: Record<string, string> = {};

  if (auth) {
    const token = getToken();
    if (token) headers['Authorization'] = `Bearer ${token}`;
  }

  let payload: BodyInit | undefined;
  if (body !== undefined) {
    if (isForm) {
      payload = body as BodyInit;
    } else {
      headers['Content-Type'] = 'application/json';
      payload = JSON.stringify(body);
    }
  }

  const response = await fetch(`${API_BASE}${path}`, { method, headers, body: payload });

  if (response.status === 204) {
    return undefined as T;
  }

  const text = await response.text();
  const data = text ? JSON.parse(text) : null;

  if (!response.ok) {
    const message = extractDetailMessage(data?.detail) ?? `Request failed with status ${response.status}`;
    throw new ApiError(response.status, message);
  }

  return data as T;
}
