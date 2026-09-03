import { NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../auth/AuthContext';

export function NavBar() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  function handleLogout() {
    logout();
    navigate('/login');
  }

  return (
    <nav className="navbar">
      <div className="navbar-brand">Phishing Detection Console</div>
      <div className="navbar-links">
        <NavLink to="/" end>
          Detect
        </NavLink>
        {user && <NavLink to="/logs">Logs</NavLink>}
        {user?.role === 'admin' && <NavLink to="/whitelist">Whitelist</NavLink>}
        {user?.role === 'admin' && <NavLink to="/metrics">Metrics</NavLink>}
      </div>
      <div className="navbar-auth">
        {user ? (
          <>
            <span className="navbar-user">
              {user.name} <span className="badge badge-role">{user.role}</span>
            </span>
            <button type="button" onClick={handleLogout}>
              Log out
            </button>
          </>
        ) : (
          <>
            <NavLink to="/login">Log in</NavLink>
            <NavLink to="/register">Sign up</NavLink>
          </>
        )}
      </div>
    </nav>
  );
}
