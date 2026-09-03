import type { Prediction } from '../api/types';

export function PredictionBadge({ prediction }: { prediction: Prediction }) {
  return <span className={`badge badge-${prediction}`}>{prediction}</span>;
}
