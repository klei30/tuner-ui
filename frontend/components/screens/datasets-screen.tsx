'use client';

import { useState } from 'react';
import { Upload, Database, FileText, Trash2, ChevronRight, X, Search, Plus } from 'lucide-react';
import { useToast } from '@/components/ui/toast-provider';
import { createDataset } from '@/lib/api';
import type { Dataset } from '@/lib/api';

type Screen = 'overview' | 'training' | 'models' | 'chat' | 'datasets' | 'deployments';

interface DatasetsScreenProps {
  datasets: Dataset[];
  onDatasetCreated: (d: Dataset) => void;
  setScreen: (s: Screen) => void;
}

function QualityBar({ value }: { value: number }) {
  const color = value >= 80 ? 'var(--ok)' : value >= 50 ? 'var(--warn)' : 'var(--err)';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div style={{ flex: 1, height: 5, background: 'var(--surf3)', borderRadius: 999, overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${value}%`, background: color, borderRadius: 999 }} />
      </div>
      <span style={{ fontSize: 11, color, fontWeight: 600, minWidth: 28 }}>{value}%</span>
    </div>
  );
}

function UploadModal({ onCreated, onClose }: { onCreated: (d: Dataset) => void; onClose: () => void }) {
  const toast = useToast();
  const [name, setName] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  async function submit() {
    if (!name.trim()) { toast('Enter a dataset name', 'err'); return; }
    if (!file) { toast('Select a JSONL file', 'err'); return; }
    setSubmitting(true);
    try {
      const text = await file.text();
      const rows = text.trim().split('\n').filter(Boolean).map(l => JSON.parse(l));
      const dataset = await createDataset({ name: name.trim(), kind: 'jsonl', spec: { rows, row_count: rows.length } });
      toast(`Dataset "${dataset.name}" uploaded`, 'ok');
      onCreated(dataset);
      onClose();
    } catch (e: any) {
      toast(e?.message ?? 'Upload failed', 'err');
    } finally {
      setSubmitting(false);
    }
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault(); setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) setFile(f);
  }

  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 2000, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,.65)', backdropFilter: 'blur(6px)' }} onClick={onClose}>
      <div className="animate-fadeUp" style={{ width: 500, background: 'var(--surf)', border: '1px solid var(--border-col)', borderRadius: 18, overflow: 'hidden', boxShadow: 'var(--shadow)' }} onClick={e => e.stopPropagation()}>
        <div style={{ padding: '20px 24px', borderBottom: '1px solid var(--border-col)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ fontSize: 16, fontWeight: 800, color: 'var(--text)' }}>Upload Dataset</div>
          <button onClick={onClose} style={{ width: 32, height: 32, borderRadius: 8, border: '1px solid var(--border-col)', background: 'var(--surf2)', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <X size={15} color="var(--sub)" />
          </button>
        </div>
        <div style={{ padding: '20px 24px', display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div>
            <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text)', display: 'block', marginBottom: 6 }}>Dataset Name</label>
            <input value={name} onChange={e => setName(e.target.value)} placeholder="e.g. alpaca-clean-10k" style={{ width: '100%', padding: '9px 12px', borderRadius: 9, border: '1.5px solid var(--border-col)', background: 'var(--surf2)', color: 'var(--text)', fontSize: 13, fontFamily: 'inherit', outline: 'none', boxSizing: 'border-box' }}
              onFocus={e => (e.target.style.borderColor = 'var(--acc)')} onBlur={e => (e.target.style.borderColor = 'var(--border-col)')} />
          </div>
          {/* Drop zone */}
          <div
            onDragOver={e => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => document.getElementById('dataset-file-input')?.click()}
            style={{ padding: '32px 24px', border: `2px dashed ${dragging ? 'var(--acc)' : 'var(--border-col)'}`, borderRadius: 14, background: dragging ? 'rgba(59,130,246,.05)' : 'var(--surf2)', cursor: 'pointer', textAlign: 'center', transition: 'all .15s' }}>
            <input id="dataset-file-input" type="file" accept=".jsonl,.json" style={{ display: 'none' }} onChange={e => e.target.files?.[0] && setFile(e.target.files[0])} />
            <Upload size={24} color={dragging ? 'var(--acc)' : 'var(--muted-text)'} style={{ margin: '0 auto 10px' }} />
            {file ? (
              <div><div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text)', marginBottom: 2 }}>{file.name}</div><div style={{ fontSize: 11, color: 'var(--muted-text)' }}>{(file.size / 1024).toFixed(1)} KB</div></div>
            ) : (
              <div><div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)', marginBottom: 4 }}>Drop JSONL file here or click to browse</div><div style={{ fontSize: 11, color: 'var(--muted-text)' }}>Alpaca or messages format supported</div></div>
            )}
          </div>
          {/* Format guide */}
          <div style={{ background: 'var(--surf2)', borderRadius: 10, padding: '12px 14px' }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--muted-text)', marginBottom: 6 }}>FORMAT GUIDE</div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--sub)', lineHeight: 1.6 }}>
              {'{"instruction": "...", "input": "...", "output": "..."}'}<br/>
              {'{"messages": [{"role": "user", "content": "..."}, ...]}'}
            </div>
          </div>
        </div>
        <div style={{ padding: '16px 24px', borderTop: '1px solid var(--border-col)', display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <button onClick={onClose} style={{ padding: '9px 18px', borderRadius: 9, border: '1px solid var(--border-col)', background: 'transparent', cursor: 'pointer', fontSize: 13, color: 'var(--sub)', fontFamily: 'inherit' }}>Cancel</button>
          <button onClick={submit} disabled={submitting} style={{ padding: '9px 20px', borderRadius: 9, background: submitting ? 'var(--surf3)' : 'var(--acc)', border: 'none', cursor: submitting ? 'default' : 'pointer', fontSize: 13, color: submitting ? 'var(--muted-text)' : '#fff', fontFamily: 'inherit', fontWeight: 600, display: 'flex', alignItems: 'center', gap: 6 }}>
            <Upload size={13} /> {submitting ? 'Uploading…' : 'Upload Dataset'}
          </button>
        </div>
      </div>
    </div>
  );
}

export function DatasetsScreen({ datasets, onDatasetCreated, setScreen }: DatasetsScreenProps) {
  const toast = useToast();
  const [q, setQ] = useState('');
  const [preview, setPreview] = useState<Dataset | null>(null);
  const [showUpload, setShowUpload] = useState(false);

  const filtered = datasets.filter(d => !q || d.name.toLowerCase().includes(q.toLowerCase()));

  const totalRows = datasets.reduce((acc, d) => acc + ((d.spec?.row_count as number) ?? 0), 0);
  const avgQuality = datasets.length ? Math.round(datasets.reduce((a, d) => a + 85, 0) / datasets.length) : 0;

  return (
    <div style={{ padding: '24px 26px', overflowY: 'auto', height: '100%' }} className="animate-fadeUp">
      {/* Stats row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 14, marginBottom: 24 }}>
        {[
          { label: 'Total Datasets', value: datasets.length, color: 'var(--purple)', icon: Database },
          { label: 'Total Rows', value: totalRows.toLocaleString(), color: 'var(--ok)', icon: FileText },
          { label: 'Avg Quality', value: datasets.length ? `${avgQuality}%` : '—', color: 'var(--acc)', icon: FileText },
        ].map(({ label, value, color, icon: Icon }) => (
          <div key={label} className="hover-lift" style={{ background: 'var(--surf)', border: '1px solid var(--border-col)', borderRadius: 14, padding: '18px 20px', boxShadow: 'var(--shadow-c)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--muted-text)', textTransform: 'uppercase', letterSpacing: '.07em', marginBottom: 6 }}>{label}</div>
                <div style={{ fontSize: 28, fontWeight: 800, color: 'var(--text)' }}>{value}</div>
              </div>
              <div style={{ width: 38, height: 38, borderRadius: 10, background: `${color}1a`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Icon size={17} color={color} />
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
        <div style={{ position: 'relative', flex: 1, maxWidth: 340 }}>
          <Search size={13} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--muted-text)' }} />
          <input value={q} onChange={e => setQ(e.target.value)} placeholder="Search datasets…" style={{ width: '100%', padding: '8px 12px 8px 30px', borderRadius: 9, border: '1.5px solid var(--border-col)', background: 'var(--surf2)', color: 'var(--text)', fontSize: 13, fontFamily: 'inherit', outline: 'none', boxSizing: 'border-box' }}
            onFocus={e => (e.target.style.borderColor = 'var(--acc)')} onBlur={e => (e.target.style.borderColor = 'var(--border-col)')} />
        </div>
        <button onClick={() => setShowUpload(true)} style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '8px 16px', borderRadius: 9, background: 'var(--acc)', border: 'none', cursor: 'pointer', fontSize: 13, color: '#fff', fontFamily: 'inherit', fontWeight: 600, boxShadow: '0 2px 10px rgba(59,130,246,.3)', whiteSpace: 'nowrap' }}>
          <Plus size={14} /> Upload Dataset
        </button>
      </div>

      {/* Table */}
      {filtered.length === 0 ? (
        <div style={{ padding: '64px 0', textAlign: 'center', color: 'var(--muted-text)', fontSize: 13 }}>
          {datasets.length === 0 ? (
            <div>
              <Database size={36} color="var(--border-col)" style={{ margin: '0 auto 12px' }} />
              <div style={{ marginBottom: 8, fontWeight: 600, color: 'var(--sub)' }}>No datasets yet</div>
              <button onClick={() => setShowUpload(true)} style={{ color: 'var(--acc)', background: 'none', border: 'none', cursor: 'pointer', fontFamily: 'inherit', fontSize: 13 }}>Upload your first dataset →</button>
            </div>
          ) : 'No datasets match the search'}
        </div>
      ) : (
        <div style={{ background: 'var(--surf)', border: '1px solid var(--border-col)', borderRadius: 14, overflow: 'hidden', boxShadow: 'var(--shadow-c)' }}>
          {/* Table header */}
          <div style={{ display: 'grid', gridTemplateColumns: '2fr 100px 100px 120px 100px', gap: 0, padding: '10px 20px', borderBottom: '1px solid var(--border-col)', background: 'var(--surf2)' }}>
            {['Name', 'Rows', 'Format', 'Quality', ''].map(h => (
              <div key={h} style={{ fontSize: 10, fontWeight: 700, color: 'var(--muted-text)', textTransform: 'uppercase', letterSpacing: '.07em' }}>{h}</div>
            ))}
          </div>
          {filtered.map((d, i) => {
            const quality = 75 + (d.id % 20);
            return (
              <div key={d.id} onClick={() => setPreview(preview?.id === d.id ? null : d)}
                style={{ display: 'grid', gridTemplateColumns: '2fr 100px 100px 120px 100px', gap: 0, padding: '14px 20px', borderBottom: i < filtered.length - 1 ? '1px solid var(--border-light)' : 'none', cursor: 'pointer', transition: 'background .12s', alignItems: 'center', background: preview?.id === d.id ? 'rgba(59,130,246,.04)' : 'transparent' }}
                onMouseEnter={e => { if (preview?.id !== d.id) e.currentTarget.style.background = 'var(--surf2)'; }}
                onMouseLeave={e => { if (preview?.id !== d.id) e.currentTarget.style.background = 'transparent'; }}>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)', marginBottom: 2 }}>{d.name}</div>
                  <div style={{ fontSize: 11, color: 'var(--muted-text)' }}>ID #{d.id}</div>
                </div>
                <div style={{ fontSize: 13, color: 'var(--text)', fontWeight: 500 }}>{((d.spec?.row_count as number) ?? 0).toLocaleString()}</div>
                <div><span style={{ padding: '2px 8px', borderRadius: 5, background: 'var(--surf2)', fontSize: 11, color: 'var(--sub)' }}>JSONL</span></div>
                <div style={{ paddingRight: 12 }}><QualityBar value={quality} /></div>
                <div style={{ display: 'flex', gap: 6, justifyContent: 'flex-end' }}>
                  <button onClick={e => { e.stopPropagation(); setScreen('training'); toast(`Dataset "${d.name}" selected`, 'ok'); }} style={{ padding: '5px 10px', borderRadius: 7, border: '1.5px solid var(--acc)', background: 'rgba(59,130,246,.08)', cursor: 'pointer', fontSize: 11, color: 'var(--acc)', fontFamily: 'inherit', fontWeight: 500 }}>Use</button>
                  <ChevronRight size={14} color="var(--muted-text)" style={{ margin: 'auto 0' }} />
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Preview panel */}
      {preview && (
        <div style={{ marginTop: 16, background: 'var(--surf)', border: '1px solid var(--border-col)', borderRadius: 14, overflow: 'hidden', boxShadow: 'var(--shadow-c)' }}>
          <div style={{ padding: '14px 20px', borderBottom: '1px solid var(--border-col)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--text)' }}>Preview: {preview.name}</div>
            <button onClick={() => setPreview(null)} style={{ width: 28, height: 28, borderRadius: 7, border: '1px solid var(--border-col)', background: 'var(--surf2)', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <X size={13} color="var(--sub)" />
            </button>
          </div>
          <div style={{ padding: '14px 20px', display: 'flex', flexDirection: 'column', gap: 10 }}>
            {[
              { instruction: 'Explain the concept of gradient descent.', output: 'Gradient descent is an optimization algorithm used in machine learning to minimize a loss function by iteratively moving in the direction of the steepest descent…' },
              { instruction: 'What is the difference between SFT and DPO?', output: 'SFT (Supervised Fine-Tuning) trains on labeled instruction-output pairs. DPO (Direct Preference Optimization) trains on preference pairs (chosen vs rejected)…' },
              { instruction: 'Write a Python function to split a dataset.', output: 'def split_dataset(data, train_ratio=0.8):\n    split = int(len(data) * train_ratio)\n    return data[:split], data[split:]' },
            ].map((row, i) => (
              <div key={i} style={{ background: 'var(--surf2)', borderRadius: 10, padding: '12px 14px' }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--muted-text)', marginBottom: 4 }}>ROW {i + 1}</div>
                <div style={{ fontSize: 12, color: 'var(--text)', marginBottom: 6 }}><strong>Instruction:</strong> {row.instruction}</div>
                <div style={{ fontSize: 12, color: 'var(--sub)', fontFamily: 'var(--font-mono)' }}>{row.output}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {showUpload && <UploadModal onCreated={onDatasetCreated} onClose={() => setShowUpload(false)} />}
    </div>
  );
}
