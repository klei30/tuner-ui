'use client';

import { createContext, useCallback, useContext, useState } from 'react';

type ToastType = 'ok' | 'err' | 'info' | 'warn';
interface Toast { id: number; msg: string; type: ToastType }

const ToastCtx = createContext<(msg: string, type?: ToastType) => void>(() => {});

const COLORS: Record<ToastType, string> = {
  ok:   '#22c55e',
  err:  '#ef4444',
  info: '#3b82f6',
  warn: '#f59e0b',
};
const ICONS: Record<ToastType, string> = { ok: '✓', err: '✕', info: 'ℹ', warn: '⚠' };

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const add = useCallback((msg: string, type: ToastType = 'ok') => {
    const id = Date.now();
    setToasts(p => [...p, { id, msg, type }]);
    setTimeout(() => setToasts(p => p.filter(x => x.id !== id)), 3000);
  }, []);

  return (
    <ToastCtx.Provider value={add}>
      {children}
      <div style={{
        position: 'fixed', bottom: 22, left: '50%', transform: 'translateX(-50%)',
        zIndex: 9999, display: 'flex', flexDirection: 'column', gap: 8, alignItems: 'center',
        pointerEvents: 'none',
      }}>
        {toasts.map(t => (
          <div key={t.id} className="animate-toastIn" style={{
            padding: '10px 18px', borderRadius: 10, background: COLORS[t.type],
            color: '#fff', fontSize: 13, fontWeight: 600,
            boxShadow: '0 8px 32px rgba(0,0,0,.3)',
            display: 'flex', alignItems: 'center', gap: 8, whiteSpace: 'nowrap',
            pointerEvents: 'auto',
          }}>
            <span>{ICONS[t.type]}</span>{t.msg}
          </div>
        ))}
      </div>
    </ToastCtx.Provider>
  );
}

export const useToast = () => useContext(ToastCtx);
