import { useEffect, useState } from 'react';
import type { FormEvent } from 'react';
import { ApiError } from '../api/client';
import { addWhitelistEntry, deleteWhitelistEntry, listWhitelist } from '../api/whitelist';
import type { WhitelistEntry } from '../api/types';
import { ListIcon, TrashIcon } from '../components/Icons';
import { ConfirmModal } from '../components/ConfirmModal';

export function WhitelistPage() {
  const [entries, setEntries] = useState<WhitelistEntry[]>([]);
  const [domain, setDomain] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<WhitelistEntry | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  function refresh() {
    setLoading(true);
    listWhitelist()
      .then(setEntries)
      .catch((err) => setError(err instanceof ApiError ? err.message : 'Failed to load whitelist.'))
      .finally(() => setLoading(false));
  }

  useEffect(refresh, []);

  async function handleAdd(event: FormEvent) {
    event.preventDefault();
    if (!domain.trim()) return;
    setError(null);
    setSubmitting(true);
    try {
      await addWhitelistEntry(domain.trim());
      setDomain('');
      refresh();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Failed to add domain.');
    } finally {
      setSubmitting(false);
    }
  }

  function openDeleteConfirm(entry: WhitelistEntry) {
    setDeleteError(null);
    setPendingDelete(entry);
  }

  async function confirmDelete() {
    if (!pendingDelete) return;
    setDeleting(true);
    setDeleteError(null);
    try {
      await deleteWhitelistEntry(pendingDelete.id);
      setEntries((prev) => prev.filter((e) => e.id !== pendingDelete.id));
      setPendingDelete(null);
    } catch (err) {
      setDeleteError(err instanceof ApiError ? err.message : 'Failed to delete domain.');
    } finally {
      setDeleting(false);
    }
  }

  return (
    <div className="page">
      <div className="page-header">
        <span className="page-header-icon">
          <ListIcon />
        </span>
        <div>
          <h1>Whitelist</h1>
          <p className="page-subtitle">Domains here bypass the classifier entirely in /detect.</p>
        </div>
      </div>

      <form className="card form-inline" onSubmit={handleAdd}>
        <input
          type="text"
          placeholder="trusted-domain.com"
          value={domain}
          onChange={(e) => setDomain(e.target.value)}
        />
        <button type="submit" disabled={submitting}>
          {submitting ? 'Adding...' : 'Add domain'}
        </button>
      </form>

      {error && <p className="page-status error">{error}</p>}
      {loading && <p className="page-status">Loading...</p>}

      {!loading && (
        <div className="table-wrapper card">
          <table>
            <thead>
              <tr>
                <th>Domain</th>
                <th>Added by (user ID)</th>
                <th>Date added</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {entries.map((entry) => (
                <tr key={entry.id}>
                  <td>{entry.domain}</td>
                  <td>{entry.added_by}</td>
                  <td>{new Date(entry.date_added).toLocaleString()}</td>
                  <td>
                    <button type="button" className="danger" onClick={() => openDeleteConfirm(entry)}>
                      Remove
                    </button>
                  </td>
                </tr>
              ))}
              {entries.length === 0 && (
                <tr>
                  <td colSpan={4} className="page-status">
                    No whitelisted domains yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      <ConfirmModal
        open={pendingDelete !== null}
        onClose={() => setPendingDelete(null)}
        onConfirm={confirmDelete}
        icon={<TrashIcon />}
        variant="danger"
        title="Remove this domain?"
        description={
          pendingDelete
            ? `"${pendingDelete.domain}" will no longer bypass the classifier once removed - future scans against it will run through the normal pipeline.`
            : undefined
        }
        confirmLabel="Remove domain"
        loading={deleting}
      >
        {deleteError && <p className="page-status error">{deleteError}</p>}
      </ConfirmModal>
    </div>
  );
}
