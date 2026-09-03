import { useEffect, useState } from 'react';
import { Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts';
import { ApiError } from '../api/client';
import { getMetrics } from '../api/metrics';
import type { Metrics } from '../api/types';
import { StatCard } from '../components/StatCard';

const COLORS = { phishing: '#e0555f', legitimate: '#3fa66a' };

function formatPercent(value: number | null): string {
  return value === null ? 'N/A' : `${(value * 100).toFixed(1)}%`;
}

export function MetricsPage() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getMetrics()
      .then(setMetrics)
      .catch((err) => setError(err instanceof ApiError ? err.message : 'Failed to load metrics.'))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <p className="page-status">Loading...</p>;
  if (error) return <p className="page-status error">{error}</p>;
  if (!metrics) return null;

  const distribution = [
    { name: 'Phishing', value: metrics.phishing_count, color: COLORS.phishing },
    { name: 'Legitimate', value: metrics.legitimate_count, color: COLORS.legitimate },
  ];

  return (
    <div className="page">
      <h1>Metrics</h1>
      <p className="page-subtitle">Operational stats reflect all logged requests. Accuracy/precision/recall/F1 only count logs with a known ground-truth label.</p>

      <div className="stat-grid">
        <StatCard label="Total requests" value={String(metrics.total_requests)} />
        <StatCard
          label="Average latency"
          value={metrics.average_latency_ms === null ? 'N/A' : `${metrics.average_latency_ms.toFixed(1)} ms`}
          hint="target: <=500ms"
        />
        <StatCard
          label="Throughput"
          value={metrics.throughput_rps === null ? 'N/A' : `${metrics.throughput_rps.toFixed(2)} req/s`}
          hint="target: >=20 req/s"
        />
        <StatCard label="Labeled sample size" value={String(metrics.labeled_sample_size)} />
      </div>

      <div className="metrics-layout">
        <div className="card chart-card">
          <h2>Prediction distribution</h2>
          {metrics.total_requests > 0 ? (
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie data={distribution} dataKey="value" nameKey="name" outerRadius={90} label>
                  {distribution.map((entry) => (
                    <Cell key={entry.name} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <p className="page-status">No requests logged yet.</p>
          )}
        </div>

        <div className="stat-grid">
          <StatCard label="Accuracy" value={formatPercent(metrics.accuracy)} hint="target: n/a" />
          <StatCard label="Precision" value={formatPercent(metrics.precision)} hint="target: >=0.91" />
          <StatCard label="Recall" value={formatPercent(metrics.recall)} hint="target: >=0.90" />
          <StatCard label="F1 score" value={formatPercent(metrics.f1_score)} hint="target: >=0.90" />
        </div>
      </div>

      {metrics.labeled_sample_size === 0 && (
        <p className="page-status">
          No labeled evaluation data yet - accuracy/precision/recall/F1 will populate once DetectionLog rows have an
          actual_label set (build order phase 5).
        </p>
      )}
    </div>
  );
}
