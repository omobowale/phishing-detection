import { useState } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext';
import { ChartIcon, HistoryIcon, ListIcon, LogOutIcon, MoonIcon, ScanIcon, ShieldIcon, SunIcon } from './Icons';
import { useTheme } from '../theme/useTheme';
import { ConfirmModal } from './ConfirmModal';

export function NavBar() {
  const { user, logout } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const navigate = useNavigate();
  const [confirmLogout, setConfirmLogout] = useState(false);

  function handleLogout() {
    setConfirmLogout(false);
    logout();
    navigate('/login');
  }

  return (
    <nav className="navbar">
      <NavLink to="/" className="navbar-brand" aria-label="PhishGuard home">
        <span className="brand-mark"><ShieldIcon /></span>
        <span>Phish<span>Guard</span></span>
      </NavLink>
      <div className="navbar-links">
        <NavLink to="/" end>
          <ScanIcon /> Scan
        </NavLink>
        {user && <NavLink to="/logs"><HistoryIcon /> History</NavLink>}
        {user?.role === 'admin' && <NavLink to="/whitelist"><ListIcon /> Allowlist</NavLink>}
        {user?.role === 'admin' && <NavLink to="/metrics"><ChartIcon /> Insights</NavLink>}
      </div>
      <div className="navbar-auth">
        <button
          type="button"
          className="icon-button theme-toggle"
          onClick={toggleTheme}
          aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {theme === 'dark' ? <SunIcon /> : <MoonIcon />}
        </button>
        {user ? (
          <>
            <span className="user-avatar">{user.name.charAt(0).toUpperCase()}</span>
            <span className="navbar-user"><strong>{user.name}</strong><small>{user.role === 'admin' ? 'Administrator' : 'Member'}</small></span>
            <button
              type="button"
              className="icon-button"
              onClick={() => setConfirmLogout(true)}
              aria-label="Log out"
              title="Log out"
            >
              <LogOutIcon />
            </button>
          </>
        ) : (
          <>
            <NavLink to="/login" className="login-link">Log in</NavLink>
            <NavLink to="/register" className="button-link">Get started</NavLink>
          </>
        )}
      </div>

      <ConfirmModal
        open={confirmLogout}
        onClose={() => setConfirmLogout(false)}
        onConfirm={handleLogout}
        icon={<LogOutIcon />}
        title="Log out of PhishGuard?"
        description="You'll need to sign in again to view your scan history and account details."
        confirmLabel="Log out"
        cancelLabel="Stay signed in"
      />
    </nav>
  );
}
