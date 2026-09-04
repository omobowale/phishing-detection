import type { ReactNode } from 'react';
import { Modal } from './Modal';

export interface ConfirmModalProps {
  open: boolean;
  onClose: () => void;
  onConfirm: () => void;
  icon?: ReactNode;
  title: string;
  description?: string;
  confirmLabel?: string;
  cancelLabel?: string;
  variant?: 'brand' | 'danger';
  loading?: boolean;
  children?: ReactNode;
}

export function ConfirmModal({
  open,
  onClose,
  onConfirm,
  icon,
  title,
  description,
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  variant = 'brand',
  loading = false,
  children,
}: ConfirmModalProps) {
  return (
    <Modal
      open={open}
      onClose={onClose}
      icon={icon}
      iconVariant={variant === 'danger' ? 'danger' : 'brand'}
      title={title}
      description={description}
      actions={
        <>
          <button type="button" className="secondary-button" onClick={onClose} disabled={loading}>
            {cancelLabel}
          </button>
          <button
            type="button"
            className={variant === 'danger' ? 'primary-button danger-button' : 'primary-button'}
            onClick={onConfirm}
            disabled={loading}
          >
            {loading && <span className="spinner" />}
            {confirmLabel}
          </button>
        </>
      }
    >
      {children}
    </Modal>
  );
}
