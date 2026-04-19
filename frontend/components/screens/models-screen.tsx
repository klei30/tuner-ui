'use client';

import { useState } from 'react';
import { Filter, Search, Cpu, Play, LayoutGrid, List } from 'lucide-react';
import { useToast } from '@/components/ui/toast-provider';
import type { SupportedModel, RegisteredModel } from '@/lib/api';

type Screen = 'overview' | 'training' | 'models' | 'chat' | 'datasets' | 'deployments';

interface ModelsScreenProps {
  supportedModels: SupportedModel[];
  registeredModels: RegisteredModel[];
  setScreen: (s: Screen) => void;
}

const ORG_COLORS: Record<string, string> = {
  'meta-llama': 'var(--acc)',
  'deepseek-ai': 'var(--blue)',
  'Qwen': 'var(--ok)',
  'mistralai': 'var(--purple)',
};

function fits16gb(params: string | null | undefined): boolean {
  if (!params) return true;
  const n = parseFloat(params);
  return n <= 8 || isNaN(n);
}

export function ModelsScreen({ supportedModels, registeredModels, setScreen }: ModelsScreenProps) {
  const toast = useToast();
  const [q, setQ] = useState('');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [orgFilter, setOrgFilter] = useState('All');
  const [sizeFilter, setSizeFilter] = useState('All');
  const [hovIdx, setHovIdx] = useState<number | null>(null);

  const orgs = ['All', ...Array.from(new Set(supportedModels.map(m => m.model_name.split('/')[0])))];
  const sizes = ['All', '1B', '3B', '8B', '70B'];

  const filtered = supportedModels.filter(m => {
    const org = m.model_name.split('/')[0];
    const name = m.model_name.split('/')[1] ?? m.model_name;
    const matchOrg = orgFilter === 'All' || org === orgFilter;
    const matchSize = sizeFilter === 'All' || (m.parameters ?? '').includes(sizeFilter.replace('B',''));
    const matchQ = !q || m.model_name.toLowerCase().includes(q.toLowerCase());
    return matchOrg && matchSize && matchQ;
  });

  return (
    <div style={{ display: 'flex', height: '100%', overflow: 'hidden' }}>
      {/* Filter sidebar */}
      <div style={{ width: 200, minWidth: 200, borderRight: '1px solid var(--border-col)', background: 'var(--surf)', padding: '20px 16px', overflowY: 'auto', flexShrink: 0 }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text)', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
          <Filter size={13} color="var(--acc)" /> Filters
        </div>

        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--muted-text)', textTransform: 'uppercase', letterSpacing: '.07em', marginBottom: 8 }}>Organization</div>
          {orgs.map(o => (
            <button key={o} onClick={() => setOrgFilter(o)} style={{ display: 'block', width: '100%', padding: '6px 10px', borderRadius: 7, border: 'none', textAlign: 'left', background: orgFilter === o ? 'rgba(59,130,246,.1)' : 'transparent', color: orgFilter === o ? 'var(--acc)' : 'var(--sub)', fontSize: 12, fontFamily: 'inherit', fontWeight: orgFilter === o ? 600 : 400, cursor: 'pointer', marginBottom: 2 }}>
              {o}
            </button>
          ))}
        </div>

        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--muted-text)', textTransform: 'uppercase', letterSpacing: '.07em', marginBottom: 8 }}>Model Size</div>
          {sizes.map(s => (
            <button key={s} onClick={() => setSizeFilter(s)} style={{ display: 'block', width: '100%', padding: '6px 10px', borderRadius: 7, border: 'none', textAlign: 'left', background: sizeFilter === s ? 'rgba(59,130,246,.1)' : 'transparent', color: sizeFilter === s ? 'var(--acc)' : 'var(--sub)', fontSize: 12, fontFamily: 'inherit', fontWeight: sizeFilter === s ? 600 : 400, cursor: 'pointer', marginBottom: 2 }}>
              {s}
            </button>
          ))}
        </div>

        <div style={{ padding: 10, background: 'rgba(59,130,246,.1)', borderRadius: 9, border: '1px solid rgba(59,130,246,.2)' }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--acc)', marginBottom: 4 }}>FREE TIER GPU</div>
          <div style={{ fontSize: 11, color: 'var(--sub)' }}>16 GB VRAM available. Models marked ✓ will fit.</div>
        </div>
      </div>

      {/* Main area */}
      <div style={{ flex: 1, overflow: 'auto', padding: '24px 26px' }} className="animate-fadeUp">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 18 }}>
          <div>
            <h1 style={{ fontSize: 20, fontWeight: 800, color: 'var(--text)' }}>Model Catalog</h1>
            <p style={{ fontSize: 13, color: 'var(--sub)', marginTop: 3 }}>{filtered.length} models available</p>
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <div style={{ display: 'flex', background: 'var(--surf2)', border: '1px solid var(--border-col)', borderRadius: 8, padding: 3 }}>
              {(['grid', 'list'] as const).map(v => (
                <button key={v} onClick={() => setViewMode(v)} style={{ padding: '5px 10px', borderRadius: 6, border: 'none', cursor: 'pointer', background: viewMode === v ? 'var(--surf)' : 'transparent', color: viewMode === v ? 'var(--acc)' : 'var(--muted-text)', boxShadow: viewMode === v ? 'var(--shadow-s)' : 'none' }}>
                  {v === 'grid' ? <LayoutGrid size={14} /> : <List size={14} />}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Search */}
        <div style={{ position: 'relative', marginBottom: 18, maxWidth: 380 }}>
          <Search size={14} style={{ position: 'absolute', left: 11, top: '50%', transform: 'translateY(-50%)', color: 'var(--muted-text)' }} />
          <input value={q} onChange={e => setQ(e.target.value)} placeholder="Search models…" style={{ width: '100%', padding: '9px 12px 9px 34px', borderRadius: 9, border: '1.5px solid var(--border-col)', background: 'var(--surf2)', color: 'var(--text)', fontSize: 13, fontFamily: 'inherit', outline: 'none' }}
            onFocus={e => (e.target.style.borderColor = 'var(--acc)')}
            onBlur={e => (e.target.style.borderColor = 'var(--border-col)')}
          />
        </div>

        {/* Registered models section */}
        {registeredModels.length > 0 && (
          <div style={{ marginBottom: 20 }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--muted-text)', textTransform: 'uppercase', letterSpacing: '.07em', marginBottom: 10 }}>Fine-tuned Models</div>
            <div style={{ display: viewMode === 'grid' ? 'grid' : 'flex', gridTemplateColumns: 'repeat(auto-fill,minmax(210px,1fr))', flexDirection: 'column' as const, gap: 12 }}>
              {registeredModels.map((m, idx) => (
                <div key={m.id} className="hover-lift" style={{ background: 'var(--surf)', border: `1px solid rgba(34,197,94,.3)`, borderRadius: 13, padding: viewMode === 'grid' ? 18 : '14px 18px', display: 'flex', flexDirection: viewMode === 'list' ? 'row' : 'column', alignItems: viewMode === 'list' ? 'center' : 'flex-start', gap: viewMode === 'list' ? 16 : 14, boxShadow: 'var(--shadow-c)', position: 'relative', overflow: 'hidden', cursor: 'pointer' }}>
                  <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 2, background: 'var(--ok)' }} />
                  <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start', flex: 1 }}>
                    <div style={{ width: 36, height: 36, borderRadius: 10, background: 'rgba(34,197,94,.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                      <Cpu size={17} color="var(--ok)" />
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: 10, color: 'var(--muted-text)', marginBottom: 1 }}>Fine-tuned</div>
                      <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text)' }}>{m.name}</div>
                      <div style={{ fontSize: 11, color: 'var(--sub)', marginTop: 2 }}>{m.base_model}</div>
                    </div>
                  </div>
                  <button onClick={() => { setScreen('chat'); }} style={{ padding: '5px 11px', borderRadius: 8, border: '1.5px solid var(--ok)', background: 'rgba(34,197,94,.1)', cursor: 'pointer', fontSize: 12, color: 'var(--ok)', fontFamily: 'inherit', fontWeight: 500, width: viewMode === 'grid' ? '100%' : 'auto', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5 }}>
                    <Play size={11} /> Test in Chat
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Supported models */}
        <div>
          {registeredModels.length > 0 && <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--muted-text)', textTransform: 'uppercase', letterSpacing: '.07em', marginBottom: 10 }}>Base Models</div>}
          <div style={{ display: viewMode === 'grid' ? 'grid' : 'flex', gridTemplateColumns: 'repeat(auto-fill,minmax(210px,1fr))', flexDirection: 'column' as const, gap: 12 }}>
            {filtered.map((m, idx) => {
              const org = m.model_name.split('/')[0];
              const name = m.model_name.split('/')[1] ?? m.model_name;
              const col = ORG_COLORS[org] ?? 'var(--purple)';
              const fit = fits16gb(m.parameters);
              const hov = hovIdx === idx;
              return (
                <div key={m.model_name} className="hover-lift" onMouseEnter={() => setHovIdx(idx)} onMouseLeave={() => setHovIdx(null)}
                  style={{ background: 'var(--surf)', border: `1px solid ${hov ? col + '50' : 'var(--border-col)'}`, borderRadius: 13, padding: viewMode === 'grid' ? 18 : '14px 18px', display: 'flex', flexDirection: viewMode === 'list' ? 'row' : 'column', alignItems: viewMode === 'list' ? 'center' : 'flex-start', gap: viewMode === 'list' ? 16 : 14, boxShadow: hov ? `0 8px 28px ${col}26` : 'var(--shadow-c)', position: 'relative', overflow: 'hidden', cursor: 'pointer' }}>
                  {hov && <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 2, background: col }} />}
                  <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start', flex: 1 }}>
                    <div style={{ width: 36, height: 36, borderRadius: 10, background: `${col}1a`, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                      <Cpu size={17} color={col} />
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: 10, color: 'var(--muted-text)', marginBottom: 1 }}>{org}</div>
                      <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text)' }}>{name}</div>
                      <div style={{ fontSize: 11, color: 'var(--sub)', marginTop: 2 }}>{m.description}</div>
                    </div>
                  </div>
                  {viewMode === 'grid' && (
                    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' as const }}>
                      {m.parameters && <span style={{ padding: '2px 7px', background: 'var(--surf2)', borderRadius: 5, fontSize: 10, color: 'var(--muted-text)' }}>{m.parameters} params</span>}
                      {m.context_length && <span style={{ padding: '2px 7px', background: 'var(--surf2)', borderRadius: 5, fontSize: 10, color: 'var(--muted-text)' }}>{(m.context_length / 1000).toFixed(0)}K ctx</span>}
                      <span style={{ padding: '2px 7px', background: fit ? 'rgba(34,197,94,.1)' : 'rgba(239,68,68,.1)', borderRadius: 5, fontSize: 10, color: fit ? 'var(--ok)' : 'var(--err)', fontWeight: 600 }}>
                        {fit ? '✓ Fits GPU' : '✗ Too large'}
                      </span>
                    </div>
                  )}
                  <button onClick={e => { e.stopPropagation(); setScreen('training'); toast(`Selected ${name} for training`, 'ok'); }}
                    style={{ padding: '5px 11px', borderRadius: 8, border: `1.5px solid ${col}`, background: `${col}1a`, cursor: 'pointer', fontSize: 12, color: col, fontFamily: 'inherit', fontWeight: 500, width: viewMode === 'grid' ? '100%' : 'auto', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5 }}>
                    <Play size={11} /> Quick Start
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
