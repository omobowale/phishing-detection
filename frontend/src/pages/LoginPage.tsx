import { useState } from 'react';
import type { FormEvent } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { ApiError } from '../api/client';
import { useAuth } from '../auth/AuthContext';
import { AlertIcon, ArrowIcon, LockIcon, ShieldIcon } from '../components/Icons';

export function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await login(email, password);
      const redirectTo = (location.state as { from?: string } | null)?.from ?? '/';
      navigate(redirectTo, { replace: true });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Login failed.');
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="page auth-page">
      <div className="auth-intro"><span className="auth-shield"><ShieldIcon /></span><div className="eyebrow"><span></span> Secure workspace</div><h1>Welcome back</h1><p>Sign in to review your scan history and security insights.</p></div>
      <form className="card form auth-card" onSubmit={handleSubmit}>
        <div className="auth-card-heading"><LockIcon /><div><h2>Sign in to PhishGuard</h2><p>Enter your account details below</p></div></div>
        <label>
          Email
          <input type="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
        </label>
        <label>
          Password
          <input type="password" required value={password} onChange={(e) => setPassword(e.target.value)} />
        </label>
        <button type="submit" className="primary-button" disabled={submitting}>
          {submitting ? 'Signing in...' : <>Sign in <ArrowIcon /></>}
        </button>
        {error && <p className="page-status error"><AlertIcon />{error}</p>}
        <p className="auth-switch">No account? <Link to="/register">Create one</Link></p>
      </form>
    </div>
  );
}
