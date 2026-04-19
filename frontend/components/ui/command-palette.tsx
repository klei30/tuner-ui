'use client';

import { useEffect, useRef, useState } from 'react';
import { Search, Home, Cpu, MessageSquare, Database, Rocket, Zap, Upload, ChevronRight } from 'lucide-react';

type Screen = 'overview' | 'training' | 'models' | 'chat' | 'datasets' | 'deployments';

interface Cmd { label: string; screen: Screen; group: string; icon: React.ElementType }

const CMDS: Cmd[] = [
  { label: 'Overview',              screen: 'overview',    group: 'Go to',  icon: Home },
  { label: 'Training Runs',         screen: 'training',    group: 'Go to',  icon: Zap },
  { label: 'Model Catalog',         screen: 'models',      group: 'Go to',  icon: Cpu },
  { label: 'Chat Playground',       screen: 'chat',        group: 'Go to',  icon: MessageSquare },
  { label: 'Datasets',              screen: 'datasets',    group: 'Go to',  icon: Database },
  { label: 'Deployments',           screen: 'deployments', group: 'Go to',  icon: Rocket },
  { label: 'Start New Training Run',screen: 'training',    group: 'Action', icon: Zap },
  { label: 'Upload Dataset',        screen: 'datasets',    group: 'Action', icon: Upload },
  { label: 'Deploy to HuggingFace', screen: 'deployments', group: 'Action', icon: Rocket },
];

interface CommandPaletteProps {
  open: boolean;
  onClose: () => void;
  setScreen: (s: Screen) => void;
}

export function CommandPalette({ open, onClose, setScreen }: CommandPaletteProps) {
  const [q, setQ] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (open) { setQ(''); setTimeout(() => inputRef.current?.focus(), 40); }
  }, [open]);

  if (!open) return null;

  const filtered = CMDS.filter(c => c.label.toLowerCase().includes(q.toLowerCase()));
  const groups = ['Go to', 'Action'];

  return (
    <div
      style={{
        position: 'fixed', inset: 0, zIndex: 2000,
        display: 'flex', alignItems: 'flex-start', justifyContent: 'center',
        paddingTop: 72,
        background: 'rgba(0,0,0,.65)', backdropFilter: 'blur(6px)',
      }}
      onClick={onClose}
    >
      <div
        className="animate-fadeUp"
        style={{
          width: 560, background: 'var(--surf)', border: '1px solid var(--border-col)',
          borderRadius: 16, overflow: 'hidden', boxShadow: 'var(--shadow)',
        }}
        onClick={e => e.stopPropagation()}
      >
        {/* Search input */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '13px 18px', borderBottom: '1px solid var(--border-col)' }}>
          <Search size={16} style={{ color: 'var(--muted-text)', flexShrink: 0 }} />
          <input
            ref={inputRef}
            value={q}
            onChange={e => setQ(e.target.value)}
            placeholder="Search screens, actions…"
            style={{
              flex: 1, background: 'none', border: 'none', outline: 'none',
              fontSize: 15, color: 'var(--text)', fontFamily: 'inherit',
            }}
          />
          <kbd style={{ padding: '2px 7px', borderRadius: 5, background: 'var(--surf2)', border: '1px solid var(--border-col)', fontSize: 11, color: 'var(--muted-text)' }}>ESC</kbd>
        </div>

        {/* Results */}
        <div style={{ maxHeight: 320, overflowY: 'auto' }}>
          {groups.map(group => {
            const items = filtered.filter(c => c.group === group);
            if (!items.length) return null;
            return (
              <div key={group}>
                <div style={{ padding: '8px 18px 4px', fontSize: 10, fontWeight: 700, color: 'var(--muted-text)', textTransform: 'uppercase', letterSpacing: '.08em' }}>{group}</div>
                {items.map(cmd => {
                  const Icon = cmd.icon;
                  return (
                    <button
                      key={cmd.label}
                      onClick={() => { setScreen(cmd.screen); onClose(); }}
                      style={{
                        display: 'flex', alignItems: 'center', gap: 12, width: '100%',
                        padding: '10px 18px', border: 'none', background: 'none',
                        cursor: 'pointer', textAlign: 'left', color: 'var(--text)',
                        fontFamily: 'inherit', fontSize: 13,
                      }}
                      onMouseEnter={e => (e.currentTarget.style.background = 'var(--surf2)')}
                      onMouseLeave={e => (e.currentTarget.style.background = 'none')}
                    >
                      <div style={{ width: 30, height: 30, borderRadius: 8, background: 'rgba(59,130,246,.1)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <Icon size={14} style={{ color: 'var(--acc)' }} />
                      </div>
                      {cmd.label}
                      <ChevronRight size={13} style={{ marginLeft: 'auto', color: 'var(--muted-text)' }} />
                    </button>
                  );
                })}
              </div>
            );
          })}
        </div>

        <div style={{ padding: '9px 18px', borderTop: '1px solid var(--border-col)', display: 'flex', gap: 14, fontSize: 11, color: 'var(--muted-text)' }}>
          <span>↵ select</span><span>ESC close</span>
        </div>
      </div>
    </div>
  );
}
