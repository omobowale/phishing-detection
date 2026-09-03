import { useState } from 'react';
import type { FormEvent } from 'react';
import { detect } from '../api/detect';
import { ApiError } from '../api/client';
import type { DetectResponse } from '../api/types';
import { PredictionBadge } from '../components/PredictionBadge';

export function DetectPage() {
  const [url, setUrl] = useState('');
  const [emailText, setEmailText] = useState('');
  const [result, setResult] = useState<DetectResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setResult(null);

    if (!url.trim() && !emailText.trim()) {
      setError("Enter a URL or email text - at least one is required.");
      return;
    }

    setSubmitting(true);
    try {
      const response = await detect({
        url: url.trim() || undefined,
        email_text: emailText.trim() || undefined,
      });
      setResult(response);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Something went wrong. Is the backend running?');
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="page">
      <h1>Check a URL or email</h1>
      <p className="page-subtitle">
        Submit a URL, email text, or both. Whitelisted domains bypass the classifier entirely.
      </p>

      <form className="card form" onSubmit={handleSubmit}>
        <label>
          URL
          <input
            type="text"
            placeholder="https://example.com/login"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
          />
        </label>

        <label>
          Email text
          <textarea
            rows={8}
            placeholder="Paste the email body (or a full .eml with headers) here..."
            value={emailText}
            onChange={(e) => setEmailText(e.target.value)}
          />
        </label>

        <button type="submit" disabled={submitting}>
          {submitting ? 'Analyzing...' : 'Analyze'}
        </button>
      </form>

      {error && <p className="page-status error">{error}</p>}

      {result && (
        <div className="card result-card">
          <div className="result-row">
            <span>Classification</span>
            <PredictionBadge prediction={result.classification} />
          </div>
          <div className="result-row">
            <span>Confidence</span>
            <strong>{(result.confidence_score * 100).toFixed(1)}%</strong>
          </div>
          <div className="result-row">
            <span>Processing time</span>
            <strong>{result.processing_time_ms.toFixed(1)} ms</strong>
          </div>
          <div className="result-row">
            <span>Whitelisted</span>
            <strong>{result.whitelisted ? 'Yes' : 'No'}</strong>
          </div>
        </div>
      )}
    </div>
  );
}
