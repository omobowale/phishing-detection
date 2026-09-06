import { useState } from 'react';
import type { FormEvent } from 'react';
import { detect } from '../api/detect';
import { ApiError } from '../api/client';
import type { DetectResponse } from '../api/types';
import { PredictionBadge } from '../components/PredictionBadge';
import { AlertIcon, ArrowIcon, CheckIcon, ClockIcon, LinkIcon, MailIcon, ScanIcon, ShieldIcon } from '../components/Icons';

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
    <div className="page detect-page">
      <section className="hero">
        <div className="eyebrow"><span></span> Rule-based threat analysis</div>
        <h1>Spot the threat.<br/><span>Before you click.</span></h1>
        <p>Analyze suspicious links and emails in seconds with intelligent, real-time phishing detection.</p>
        <div className="trust-row"><span><CheckIcon /> Private analysis</span><span><CheckIcon /> Instant results</span><span><CheckIcon /> No data shared</span></div>
      </section>

      <section className="scanner-card">
        <div className="scanner-heading">
          <span className="scanner-icon"><ScanIcon /></span>
          <div><h2>Threat scanner</h2><p>Paste anything suspicious below</p></div>
          <span className="status-pill"><i></i> System operational</span>
        </div>
        <form className="form scanner-form" onSubmit={handleSubmit}>
        <label className="field-label">
          <span><LinkIcon /> Website URL <small>Optional</small></span>
          <input
            type="text"
            placeholder="https://suspicious-website.com/login"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
          />
        </label>

        <div className="input-divider"><span>or analyze email content</span></div>
        <label className="field-label">
          <span><MailIcon /> Email content <small>Optional</small></span>
          <textarea
            rows={6}
            placeholder="Paste the suspicious email message, including headers if available..."
            value={emailText}
            onChange={(e) => setEmailText(e.target.value)}
          />
          <em>{emailText.length.toLocaleString()} characters</em>
        </label>

        <button type="submit" className="primary-button scan-button" disabled={submitting}>
          {submitting ? <><span className="spinner" /> Analyzing threat...</> : <><ShieldIcon /> Analyze for threats <ArrowIcon /></>}
        </button>
      </form>
      </section>

      {error && <p className="page-status error"><AlertIcon /> {error}</p>}

      {result && (
        <section className={`result-card result-${result.classification}`}>
          <div className="result-summary">
            <span className="result-icon">{result.classification === 'phishing' ? <AlertIcon /> : <ShieldIcon />}</span>
            <div><span className="result-kicker">Analysis complete</span><h2>{result.classification === 'phishing' ? 'Potential threat detected' : 'Looks safe'}</h2><p>{result.classification === 'phishing' ? 'This content shows patterns commonly associated with phishing.' : 'No significant phishing indicators were detected.'}</p></div>
            <PredictionBadge prediction={result.classification} />
          </div>
          <div className="result-stats">
            <div><span>Confidence</span><strong>{(result.confidence_score * 100).toFixed(1)}%</strong><div className="confidence-bar"><i style={{width: `${result.confidence_score * 100}%`}} /></div></div>
            <div><span><ClockIcon /> Processing time</span><strong>{result.processing_time_ms.toFixed(1)} ms</strong></div>
            <div><span><CheckIcon /> Trusted source</span><strong>{result.whitelisted ? 'Allowlisted' : 'Not allowlisted'}</strong></div>
          </div>
        </section>
      )}

      <section className="feature-strip">
        <div><span><ShieldIcon /></span><div><strong>Heuristic detection</strong><p>Weighted rule-based scoring (ML model in development)</p></div></div>
        <div><span><ClockIcon /></span><div><strong>Real-time results</strong><p>Threat checks in milliseconds</p></div></div>
        <div><span><MailIcon /></span><div><strong>URL & email analysis</strong><p>One scanner for every message</p></div></div>
      </section>
    </div>
  );
}
