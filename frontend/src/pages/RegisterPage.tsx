import { useState } from 'react';
import type { FormEvent } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { ApiError } from '../api/client';
import { useAuth } from '../auth/AuthContext';
import { AlertIcon, ArrowIcon, ShieldIcon } from '../components/Icons';

export function RegisterPage() {
  const { register } = useAuth();
  const navigate = useNavigate();
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await register(name, email, password);
      navigate('/', { replace: true });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Registration failed.');
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="page auth-page">
      <div className="auth-intro"><span className="auth-shield"><ShieldIcon /></span><div className="eyebrow"><span></span> Safer browsing starts here</div><h1>Create your account</h1><p>Save your scan history and keep suspicious content under control.</p></div>
      <form className="card form auth-card" onSubmit={handleSubmit}>
        <div className="auth-card-heading"><ShieldIcon /><div><h2>Join PhishGuard</h2><p>Free, secure, and ready in seconds</p></div></div>
        <label>
          Name
          <input type="text" required value={name} onChange={(e) => setName(e.target.value)} />
        </label>
        <label>
          Email
          <input type="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
        </label>
        <label>
          Password
          <input
            type="password"
            required
            minLength={8}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </label>
        <button type="submit" className="primary-button" disabled={submitting}>
          {submitting ? 'Creating account...' : <>Create account <ArrowIcon /></>}
        </button>
        {error && <p className="page-status error"><AlertIcon />{error}</p>}
        <p className="auth-switch">Already have an account? <Link to="/login">Sign in</Link></p>
      </form>
    </div>
  );
}
