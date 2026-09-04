import { useEffect, useRef } from 'react';
import type { ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { CloseIcon } from './Icons';

export interface ModalProps {
  open: boolean;
  onClose: () => void;
  icon?: ReactNode;
  iconVariant?: 'brand' | 'danger' | 'success';
  title: string;
  description?: string;
  children?: ReactNode;
  actions?: ReactNode;
}

export function Modal({ open, onClose, icon, iconVariant = 'brand', title, description, children, actions }: ModalProps) {
  const cardRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') onClose();
    }

    document.addEventListener('keydown', handleKeyDown);
    document.body.style.overflow = 'hidden';
    cardRef.current?.focus();

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = '';
    };
  }, [open, onClose]);

  if (!open) return null;

  return createPortal(
    <div
      className="modal-overlay"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div
        className="modal-card"
        role="dialog"
        aria-modal="true"
        aria-labelledby="modal-title"
        tabIndex={-1}
        ref={cardRef}
      >
        <button type="button" className="modal-close" onClick={onClose} aria-label="Close">
          <CloseIcon />
        </button>

        {icon && <span className={`modal-icon modal-icon-${iconVariant}`}>{icon}</span>}
        <h2 id="modal-title">{title}</h2>
        {description && <p className="modal-description">{description}</p>}
        {children && <div className="modal-content">{children}</div>}
        {actions && <div className="modal-actions">{actions}</div>}
      </div>
    </div>,
    document.body,
  );
}
