'use client';

import { useState, useEffect } from 'react';
import { Menu, Search, Bell, Sun, Moon, ChevronDown, Check, Plus } from 'lucide-react';
import { useTheme } from 'next-themes';
import type { Project } from '@/lib/api';

interface HeaderProps {
  collapsed: boolean;
  setCollapsed: (v: boolean) => void;
  projects: Project[];
  selectedProjectId: number | null;
  onSelectProject: (id: number) => void;
  onCreateProject: () => void;
  onCmd: () => void;
}

export function Header({ collapsed, setCollapsed, projects, selectedProjectId, onSelectProject, onCreateProject, onCmd }: HeaderProps) {
  const { resolvedTheme, setTheme } = useTheme();
  const [dropOpen, setDropOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  const dark = resolvedTheme === 'dark';
  const selectedProject = projects.find(p => p.id === selectedProjectId);

  return (
    <div style={{
      height: 54, background: 'var(--surf)',
      borderBottom: '1px solid var(--border-col)',
      display: 'flex', alignItems: 'center',
      padding: '0 12px', gap: 0, flexShrink: 0, zIndex: 20,
    }}>
      {/* Hamburger */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 6, borderRadius: 7, display: 'flex' }}
        onMouseEnter={e => (e.currentTarget.style.background = 'var(--surf2)')}
        onMouseLeave={e => (e.currentTarget.style.background = 'none')}
      >
        <Menu size={17} color="var(--sub)" />
      </button>

      <div style={{ width: 1, height: 20, background: 'var(--border-col)' }} />
      <span style={{ fontSize: 11, color: 'var(--muted-text)', fontWeight: 600, letterSpacing: '.05em' }}>PROJECT</span>

      {/* Project selector */}
      <div style={{ position: 'relative' }}>
        <button
          onClick={() => setDropOpen(v => !v)}
          style={{
            display: 'flex', alignItems: 'center', gap: 6,
            padding: '5px 12px', borderRadius: 8,
            background: 'var(--surf2)', border: '1px solid var(--border-col)',
            cursor: 'pointer', fontFamily: 'inherit', fontSize: 13, fontWeight: 500, color: 'var(--text)',
          }}
        >
          <div style={{ width: 6, height: 6, borderRadius: 999, background: 'var(--ok)' }} />
          {selectedProject?.name ?? 'Select project'}
          <ChevronDown size={12} color="var(--sub)" />
        </button>

        {dropOpen && (
          <div className="animate-fadeUp" style={{
            position: 'absolute', top: 'calc(100% + 6px)', left: 0,
            background: 'var(--surf)', border: '1px solid var(--border-col)',
            borderRadius: 12, boxShadow: 'var(--shadow)', zIndex: 200,
            minWidth: 190, overflow: 'hidden',
          }}>
            <div style={{ padding: '8px 14px 4px', fontSize: 10, fontWeight: 700, color: 'var(--muted-text)', textTransform: 'uppercase', letterSpacing: '.07em' }}>Your Projects</div>
            {projects.map(p => (
              <button
                key={p.id}
                onClick={() => { onSelectProject(p.id); setDropOpen(false); }}
                style={{
                  display: 'flex', alignItems: 'center', gap: 8,
                  width: '100%', padding: '9px 14px', background: 'none', border: 'none',
                  cursor: 'pointer', fontSize: 13,
                  color: p.id === selectedProjectId ? 'var(--acc)' : 'var(--text)',
                  fontFamily: 'inherit', fontWeight: p.id === selectedProjectId ? 600 : 400,
                }}
                onMouseEnter={e => (e.currentTarget.style.background = 'var(--surf2)')}
                onMouseLeave={e => (e.currentTarget.style.background = 'none')}
              >
                {p.id === selectedProjectId ? <Check size={13} color="var(--acc)" /> : <div style={{ width: 13 }} />}
                {p.name}
              </button>
            ))}
            <div style={{ borderTop: '1px solid var(--border-col)', margin: '4px 0' }} />
            <button
              onClick={() => { onCreateProject(); setDropOpen(false); }}
              style={{
                display: 'flex', alignItems: 'center', gap: 8,
                width: '100%', padding: '9px 14px 12px', background: 'none', border: 'none',
                cursor: 'pointer', fontSize: 13, color: 'var(--acc)', fontFamily: 'inherit', fontWeight: 500,
              }}
              onMouseEnter={e => (e.currentTarget.style.background = 'var(--surf2)')}
              onMouseLeave={e => (e.currentTarget.style.background = 'none')}
            >
              <Plus size={13} color="var(--acc)" /> New Project
            </button>
          </div>
        )}
      </div>

      {/* Right side */}
      <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6 }}>
        {/* ⌘K */}
        <button
          onClick={onCmd}
          style={{
            display: 'flex', alignItems: 'center', gap: 6,
            padding: '5px 12px', borderRadius: 8,
            background: 'var(--surf2)', border: '1px solid var(--border-col)',
            cursor: 'pointer', fontSize: 12, color: 'var(--muted-text)', fontFamily: 'inherit',
          }}
          onMouseEnter={e => (e.currentTarget.style.borderColor = 'var(--acc)')}
          onMouseLeave={e => (e.currentTarget.style.borderColor = 'var(--border-col)')}
        >
          <Search size={13} color="var(--muted-text)" />
          <span>Quick search</span>
          <kbd style={{ padding: '1px 5px', borderRadius: 4, background: 'var(--surf)', border: '1px solid var(--border-col)', fontSize: 10 }}>⌘K</kbd>
        </button>

        {/* Bell */}
        <button style={{ width: 34, height: 34, borderRadius: 8, background: 'var(--surf2)', border: '1px solid var(--border-col)', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
          <Bell size={15} color="var(--sub)" />
          <div style={{ position: 'absolute', top: 7, right: 7, width: 6, height: 6, borderRadius: 999, background: 'var(--err)', border: '1.5px solid var(--surf)' }} />
        </button>

        {/* Dark/light toggle */}
        {mounted && (
          <button
            onClick={() => setTheme(dark ? 'light' : 'dark')}
            style={{ width: 34, height: 34, borderRadius: 8, background: 'var(--surf2)', border: '1px solid var(--border-col)', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
          >
            {dark ? <Sun size={15} color="var(--sub)" /> : <Moon size={15} color="var(--sub)" />}
          </button>
        )}
      </div>
    </div>
  );
}
