export type Prediction = 'phishing' | 'legitimate';
export type InputType = 'url' | 'email_text' | 'both';
export type UserRole = 'admin' | 'end_user';

export interface User {
  user_id: number;
  name: string;
  email: string;
  role: UserRole;
}

export interface DetectRequest {
  url?: string;
  email_text?: string;
}

export interface DetectResponse {
  classification: Prediction;
  confidence_score: number;
  processing_time_ms: number;
  whitelisted: boolean;
}

export interface WhitelistEntry {
  id: number;
  domain: string;
  added_by: number;
  date_added: string;
}

export interface DetectionLogEntry {
  id: number;
  user_id: number | null;
  input_type: InputType;
  input_data: string;
  prediction: Prediction;
  confidence_score: number;
  processing_time: number;
  timestamp: string;
  actual_label: Prediction | null;
}

export interface PaginatedLogs {
  total: number;
  page: number;
  page_size: number;
  items: DetectionLogEntry[];
}

export interface Metrics {
  total_requests: number;
  phishing_count: number;
  legitimate_count: number;
  average_latency_ms: number | null;
  throughput_rps: number | null;
  labeled_sample_size: number;
  accuracy: number | null;
  precision: number | null;
  recall: number | null;
  f1_score: number | null;
}

export interface LogFilters {
  page?: number;
  page_size?: number;
  classification?: Prediction;
  user_id?: number;
  start_date?: string;
  end_date?: string;
}
