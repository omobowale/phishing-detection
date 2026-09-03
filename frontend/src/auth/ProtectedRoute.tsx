import type { ReactNode } from 'react';
import { Link, Navigate } from 'react-router-dom';
import { useAuth } from './AuthContext';
import { LockIcon } from '../components/Icons';

export function ProtectedRoute({ children, adminOnly = false }: { children: ReactNode; adminOnly?: boolean }) {
  const { user, loading } = useAuth();

  if (loading) return <p className="page-status">Loading...</p>;
  if (!user) return <Navigate to="/login" replace />;
  if (adminOnly && user.role !== 'admin') {
    return (
      <div className="empty-state">
        <span className="empty-state-icon">
          <LockIcon />
        </span>
        <h2>Admin access required</h2>
        <p>This area is restricted to administrator accounts. If you think this is a mistake, contact your workspace admin.</p>
        <Link to="/" className="primary-button">
          Back to Scan
        </Link>
      </div>
    );
  }
  return <>{children}</>;
}
