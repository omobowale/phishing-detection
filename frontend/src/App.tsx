import { Navigate, Route, Routes } from 'react-router-dom';
import { NavBar } from './components/NavBar';
import { ProtectedRoute } from './auth/ProtectedRoute';
import { DetectPage } from './pages/DetectPage';
import { LoginPage } from './pages/LoginPage';
import { RegisterPage } from './pages/RegisterPage';
import { LogsPage } from './pages/LogsPage';
import { WhitelistPage } from './pages/WhitelistPage';
import { MetricsPage } from './pages/MetricsPage';

function App() {
  return (
    <div className="app-shell">
      <NavBar />
      <main>
        <Routes>
          <Route path="/" element={<DetectPage />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
          <Route
            path="/logs"
            element={
              <ProtectedRoute>
                <LogsPage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/whitelist"
            element={
              <ProtectedRoute adminOnly>
                <WhitelistPage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/metrics"
            element={
              <ProtectedRoute adminOnly>
                <MetricsPage />
              </ProtectedRoute>
            }
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
      <footer><span>PhishGuard AI</span><span className="footer-dot">•</span><span>Threat intelligence, simplified.</span></footer>
    </div>
  );
}

export default App;
