import { useEffect, useState } from 'react';
import { ApiError } from '../api/client';
import { listLogs } from '../api/logs';
import type { DetectionLogEntry, Prediction } from '../api/types';
import { PredictionBadge } from '../components/PredictionBadge';
import { useAuth } from '../auth/AuthContext';
import { HistoryIcon } from '../components/Icons';

const PAGE_SIZE = 20;

export function LogsPage() {
  const { user } = useAuth();
  const [logs, setLogs] = useState<DetectionLogEntry[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [classification, setClassification] = useState<Prediction | ''>('');
  const [userIdFilter, setUserIdFilter] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    listLogs({
      page,
      page_size: PAGE_SIZE,
      classification: classification || undefined,
      user_id: user?.role === 'admin' && userIdFilter ? Number(userIdFilter) : undefined,
      start_date: startDate ? new Date(startDate).toISOString() : undefined,
      end_date: endDate ? new Date(endDate).toISOString() : undefined,
    })
      .then((data) => {
        if (cancelled) return;
        setLogs(data.items);
        setTotal(data.total);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err instanceof ApiError ? err.message : 'Failed to load logs.');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [page, classification, userIdFilter, startDate, endDate, user?.role]);

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <div className="page">
      <div className="page-header">
        <span className="page-header-icon">
          <HistoryIcon />
        </span>
        <div>
          <h1>Detection logs</h1>
          <p className="page-subtitle">
            {user?.role === 'admin' ? 'Showing all logs.' : 'Showing your own submissions.'}
          </p>
        </div>
      </div>

      <div className="filters card">
        <label>
          Classification
          <select
            value={classification}
            onChange={(e) => {
              setPage(1);
              setClassification(e.target.value as Prediction | '');
            }}
          >
            <option value="">All</option>
            <option value="phishing">Phishing</option>
            <option value="legitimate">Legitimate</option>
          </select>
        </label>

        {user?.role === 'admin' && (
          <label>
            User ID
            <input
              type="number"
              placeholder="Any"
              value={userIdFilter}
              onChange={(e) => {
                setPage(1);
                setUserIdFilter(e.target.value);
              }}
            />
          </label>
        )}

        <label>
          From
          <input
            type="date"
            value={startDate}
            onChange={(e) => {
              setPage(1);
              setStartDate(e.target.value);
            }}
          />
        </label>

        <label>
          To
          <input
            type="date"
            value={endDate}
            onChange={(e) => {
              setPage(1);
              setEndDate(e.target.value);
            }}
          />
        </label>
      </div>

      {error && <p className="page-status error">{error}</p>}
      {loading && <p className="page-status">Loading...</p>}

      {!loading && !error && (
        <>
          <div className="table-wrapper card">
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Type</th>
                  <th>Input</th>
                  <th>Prediction</th>
                  <th>Confidence</th>
                  <th>Latency (ms)</th>
                  <th>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {logs.map((log) => (
                  <tr key={log.id}>
                    <td>{log.id}</td>
                    <td>{log.input_type}</td>
                    <td className="truncate" title={log.input_data}>
                      {log.input_data}
                    </td>
                    <td>
                      <PredictionBadge prediction={log.prediction} />
                    </td>
                    <td>{(log.confidence_score * 100).toFixed(1)}%</td>
                    <td>{log.processing_time.toFixed(1)}</td>
                    <td>{new Date(log.timestamp).toLocaleString()}</td>
                  </tr>
                ))}
                {logs.length === 0 && (
                  <tr>
                    <td colSpan={7} className="page-status">
                      No logs found.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          <div className="pagination">
            <button type="button" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
              Previous
            </button>
            <span>
              Page {page} of {totalPages} ({total} total)
            </span>
            <button type="button" disabled={page >= totalPages} onClick={() => setPage((p) => p + 1)}>
              Next
            </button>
          </div>
        </>
      )}
    </div>
  );
}
