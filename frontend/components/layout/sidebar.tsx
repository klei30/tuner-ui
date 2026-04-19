'use client';

import { Home, Zap, Cpu, MessageSquare, Database, Rocket, Settings, Search } from 'lucide-react';

type Screen = 'overview' | 'training' | 'models' | 'chat' | 'datasets' | 'deployments';

const NAV: { id: Screen; label: string; Icon: React.ElementType }[] = [
  { id: 'overview',    label: 'Overview',    Icon: Home },
  { id: 'training',    label: 'Training',    Icon: Zap },
  { id: 'models',      label: 'Models',      Icon: Cpu },
  { id: 'chat',        label: 'Chat',        Icon: MessageSquare },
  { id: 'datasets',    label: 'Datasets',    Icon: Database },
  { id: 'deployments', label: 'Deployments', Icon: Rocket },
];

interface SidebarProps {
  screen: Screen;
  setScreen: (s: Screen) => void;
  collapsed: boolean;
  onCmd: () => void;
  activeRunCount?: number;
}

export function Sidebar({ screen, setScreen, collapsed, onCmd, activeRunCount = 0 }: SidebarProps) {
  const w = collapsed ? 62 : 218;

  return (
    <div style={{
      width: w, minWidth: w,
      background: 'var(--surf)',
      borderRight: '1px solid var(--border-col)',
      display: 'flex', flexDirection: 'column',
      transition: 'width .22s cubic-bezier(.4,0,.2,1)',
      overflow: 'hidden', zIndex: 10, flexShrink: 0,
    }}>
      {/* Logo */}
      <div style={{
        padding: collapsed ? '16px 0' : '14px 12px',
        display: 'flex', alignItems: 'center', gap: 10,
        borderBottom: '1px solid var(--border-col)',
        justifyContent: collapsed ? 'center' : 'flex-start',
        flexShrink: 0,
      }}>
        <img
          src="/favicon.png" alt="Tuner UI"
          style={{
            width: 38, height: 38, borderRadius: 10, flexShrink: 0,
            boxShadow: '0 2px 8px rgba(0,0,0,.15)',
          }}
        />
        {!collapsed && (
          <div>
            <div style={{ fontSize: 14, fontWeight: 800, color: 'var(--text)', lineHeight: 1, letterSpacing: '-0.02em' }}>Tuner</div>
            <div style={{ fontSize: 10, color: 'var(--muted-text)', marginTop: 1 }}>Fine-tune LLMs</div>
          </div>
        )}
      </div>

      {/* ⌘K search shortcut */}
      {!collapsed && (
        <button
          onClick={onCmd}
          style={{
            margin: '10px 10px 4px',
            display: 'flex', alignItems: 'center', gap: 8,
            padding: '7px 10px', borderRadius: 8,
            border: '1px solid var(--border-col)', background: 'var(--surf2)',
            cursor: 'pointer', color: 'var(--muted-text)',
            fontFamily: 'inherit', fontSize: 12,
            transition: 'all .15s',
          }}
          onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--acc)'; e.currentTarget.style.color = 'var(--acc)'; }}
          onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border-col)'; e.currentTarget.style.color = 'var(--muted-text)'; }}
        >
          <Search size={13} />
          <span>Search…</span>
          <kbd style={{ marginLeft: 'auto', padding: '1px 5px', borderRadius: 4, background: 'var(--surf)', border: '1px solid var(--border-col)', fontSize: 10 }}>⌘K</kbd>
        </button>
      )}

      {/* Nav */}
      <nav style={{ padding: collapsed ? '12px 6px' : '8px 10px', flex: 1, display: 'flex', flexDirection: 'column', gap: 2 }}>
        {NAV.map(item => {
          const active = screen === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setScreen(item.id)}
              title={collapsed ? item.label : undefined}
              style={{
                display: 'flex', alignItems: 'center', gap: 10,
                padding: collapsed ? '10px' : '9px 12px',
                borderRadius: 9, border: 'none', cursor: 'pointer',
                fontFamily: 'inherit', fontSize: 13,
                fontWeight: active ? 600 : 400,
                background: active ? 'rgba(59,130,246,.1)' : 'transparent',
                color: active ? 'var(--acc)' : 'var(--sub)',
                transition: 'all .15s',
                justifyContent: collapsed ? 'center' : 'flex-start',
                width: '100%', position: 'relative',
              }}
              onMouseEnter={e => { if (!active) e.currentTarget.style.background = 'var(--surf2)'; }}
              onMouseLeave={e => { if (!active) e.currentTarget.style.background = 'transparent'; }}
            >
              <item.Icon size={17} color={active ? 'var(--acc)' : 'var(--sub)'} />
              {!collapsed && <span>{item.label}</span>}
              {item.id === 'training' && activeRunCount > 0 && !collapsed && (
                <span style={{ marginLeft: 'auto', padding: '1px 6px', borderRadius: 999, fontSize: 10, fontWeight: 700, background: 'rgba(59,130,246,.15)', color: 'var(--acc)' }}>
                  {activeRunCount}
                </span>
              )}
              {active && !collapsed && <div style={{ marginLeft: item.id === 'training' && activeRunCount > 0 ? 0 : 'auto', width: 5, height: 5, borderRadius: 999, background: 'var(--acc)' }} />}
            </button>
          );
        })}
      </nav>

      {/* User profile */}
      <div style={{ borderTop: '1px solid var(--border-col)', padding: collapsed ? '12px 6px' : '12px 14px', flexShrink: 0 }}>
        {!collapsed ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{ width: 32, height: 32, borderRadius: 999, background: 'linear-gradient(135deg,var(--acc),#8b5cf6)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
              <span style={{ fontSize: 13, fontWeight: 700, color: '#fff' }}>T</span>
            </div>
            <div>
              <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text)' }}>Tuner UI</div>
              <div style={{ fontSize: 11, color: 'var(--muted-text)' }}>Self-hosted</div>
            </div>
            <Settings size={14} color="var(--muted-text)" style={{ marginLeft: 'auto', cursor: 'pointer' }} />
          </div>
        ) : (
          <div style={{ width: 32, height: 32, borderRadius: 999, background: 'linear-gradient(135deg,var(--acc),#8b5cf6)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto' }}>
            <span style={{ fontSize: 13, fontWeight: 700, color: '#fff' }}>T</span>
          </div>
        )}
      </div>
    </div>
  );
}
