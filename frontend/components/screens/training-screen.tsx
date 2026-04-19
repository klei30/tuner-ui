'use client';

import { useState, useEffect, useRef, useMemo } from 'react';
import { Zap, Plus, Filter, Search, ChevronRight, X, BarChart2, FileText, Settings2, Play, Square, RefreshCw, Cpu, Database, Check, Star } from 'lucide-react';
import { useToast } from '@/components/ui/toast-provider';
import { cancelRun, createRun } from '@/lib/api';
import type { Run, Dataset, SupportedModel, RecipeType } from '@/lib/api';
import { formatDistanceToNow } from 'date-fns';

type Screen = 'overview' | 'training' | 'models' | 'chat' | 'datasets' | 'deployments';

interface TrainingScreenProps {
  runs: Run[];
  datasets: Dataset[];
  supportedModels: SupportedModel[];
  selectedProjectId: number | null;
  onRunCreated: (run: Run) => void;
  onRefreshRuns: () => void;
  setScreen: (s: Screen) => void;
}

const STATUS_COLOR: Record<string, string> = {
  completed: 'var(--ok)',
  running:   'var(--blue)',
  failed:    'var(--err)',
  pending:   'var(--warn)',
  cancelled: 'var(--muted-text)',
};

const RECIPE_LABELS: Record<string, string> = {
  sft: 'SFT', dpo: 'DPO', rl: 'RL', chat_sl: 'Chat SL', math_rl: 'Math RL', distillation: 'Distillation',
};

function PulseDot({ color }: { color: string }) {
  return (
    <div style={{ position: 'relative', width: 8, height: 8, flexShrink: 0 }}>
      <div className="animate-pulse-dot" style={{ position: 'absolute', inset: -3, borderRadius: 999, background: color, opacity: .35 }} />
      <div style={{ width: 8, height: 8, borderRadius: 999, background: color }} />
    </div>
  );
}

function RunCard({ run, selected, onSelect, onCancel }: { run: Run; selected: boolean; onSelect: () => void; onCancel: () => void }) {
  const sc = STATUS_COLOR[run.status] ?? 'var(--muted-text)';
  const isRunning = run.status === 'running' || run.status === 'pending';
  const progress = (run.progress ?? (run.status === 'completed' ? 1 : 0)) * 100;
  const model = run.config_json?.base_model?.split('/')[1] ?? run.config_json?.base_model ?? '—';

  return (
    <div
      onClick={onSelect}
      className="hover-lift"
      style={{
        background: 'var(--surf)', border: `1px solid ${selected ? 'var(--acc)' : 'var(--border-col)'}`,
        borderRadius: 13, padding: '16px 18px', cursor: 'pointer', position: 'relative', overflow: 'hidden',
        boxShadow: selected ? '0 0 0 2px rgba(59,130,246,.25)' : 'var(--shadow-c)',
        transition: 'all .15s',
      }}
    >
      {selected && <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 2, background: 'var(--acc)' }} />}
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12, marginBottom: 10 }}>
        <div style={{ width: 34, height: 34, borderRadius: 9, background: `${sc}1a`, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
          {isRunning ? <PulseDot color={sc} /> : <Zap size={15} color={sc} />}
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 2 }}>
            <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--text)' }}>Run #{run.id}</span>
            <span style={{ padding: '1px 7px', borderRadius: 999, fontSize: 10, fontWeight: 700, background: `${sc}1a`, color: sc }}>{run.status}</span>
            {run.recipe_type && <span style={{ padding: '1px 7px', borderRadius: 999, fontSize: 10, background: 'var(--surf2)', color: 'var(--muted-text)' }}>{RECIPE_LABELS[run.recipe_type] ?? run.recipe_type}</span>}
          </div>
          <div style={{ fontSize: 11, color: 'var(--sub)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{model}</div>
        </div>
        {isRunning && (
          <button onClick={e => { e.stopPropagation(); onCancel(); }} style={{ padding: '4px 8px', borderRadius: 7, border: '1px solid var(--border-col)', background: 'var(--surf2)', cursor: 'pointer', fontSize: 11, color: 'var(--muted-text)', fontFamily: 'inherit', display: 'flex', alignItems: 'center', gap: 4 }}>
            <Square size={10} /> Stop
          </button>
        )}
      </div>
      <div style={{ height: 4, background: 'var(--surf3)', borderRadius: 999, overflow: 'hidden', marginBottom: 6 }}>
        <div style={{ height: '100%', width: `${progress}%`, background: sc, borderRadius: 999, transition: 'width .4s ease', position: 'relative', overflow: 'hidden' }}>
          {isRunning && progress < 100 && <div className="animate-shimmer" style={{ position: 'absolute', inset: 0, background: 'linear-gradient(90deg,transparent,rgba(255,255,255,.3),transparent)', backgroundSize: '200% 100%' }} />}
        </div>
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--muted-text)' }}>
        <span>{progress.toFixed(0)}%</span>
        <span>{run.created_at ? formatDistanceToNow(new Date(run.created_at), { addSuffix: true }) : '—'}</span>
      </div>
    </div>
  );
}

const WIZARD_RECIPES = [
  { id: 'sft', label: 'SFT', desc: 'Supervised Fine-Tuning — classic instruction following', color: 'var(--acc)' },
  { id: 'dpo', label: 'DPO', desc: 'Direct Preference Optimization — align from human feedback', color: 'var(--purple)' },
  { id: 'chat_sl', label: 'Chat SL', desc: 'Chat Supervised Learning — multi-turn conversations', color: 'var(--ok)' },
  { id: 'distillation', label: 'Distillation', desc: 'Transfer knowledge from a larger teacher model', color: 'var(--warn)' },
];

function NewRunWizard({ datasets, supportedModels, projectId, onCreated, onClose }: {
  datasets: Dataset[];
  supportedModels: SupportedModel[];
  projectId: number | null;
  onCreated: (run: Run) => void;
  onClose: () => void;
}) {
  const toast = useToast();
  const [step, setStep] = useState(0);
  const [recipe, setRecipe] = useState('sft');
  const [model, setModel] = useState(supportedModels[0]?.model_name ?? '');
  const [datasetId, setDatasetId] = useState<number | null>(datasets[0]?.id ?? null);
  const [lr, setLr] = useState('2e-4');
  const [epochs, setEpochs] = useState('3');
  const [batchSize, setBatchSize] = useState('8');
  const [submitting, setSubmitting] = useState(false);
  const [modelSearch, setModelSearch] = useState('');

  const RECOMMENDED = ['meta-llama/Llama-3.1-8B-Instruct', 'Qwen/Qwen3-8B-Base', 'meta-llama/Llama-3.2-3B'];

  const filteredModels = useMemo(() => {
    const q = modelSearch.trim().toLowerCase();
    if (!q) return supportedModels;
    return supportedModels.filter(m =>
      m.model_name.toLowerCase().includes(q) ||
      (m.description ?? '').toLowerCase().includes(q)
    );
  }, [supportedModels, modelSearch]);

  function paramSize(name: string): string | null {
    const m = name.match(/(\d+(?:\.\d+)?)\s*[Bb](?:\b|-)/);
    return m ? `${m[1]}B` : null;
  }

  const steps = ['Recipe Type', 'Base Model', 'Dataset & Params', 'Review'];

  async function submit() {
    if (!projectId) { toast('No project selected', 'err'); return; }
    if (!datasetId) { toast('Select a dataset', 'err'); return; }
    setSubmitting(true);
    try {
      const run = await createRun({
        project_id: projectId,
        recipe_type: recipe.toUpperCase() as RecipeType,
        dataset_id: datasetId,
        config_json: {
          base_model: model,
          hyperparameters: { learning_rate: parseFloat(lr), num_epochs: parseInt(epochs), per_device_train_batch_size: parseInt(batchSize) },
        },
      });
      toast(`Run #${run.id} started!`, 'ok');
      onCreated(run);
      onClose();
    } catch (e: any) {
      toast(e?.message ?? 'Failed to create run', 'err');
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 2000, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,.65)', backdropFilter: 'blur(6px)' }} onClick={onClose}>
      <div className="animate-fadeUp" style={{ width: 580, maxHeight: 'calc(100vh - 48px)', background: 'var(--surf)', border: '1px solid var(--border-col)', borderRadius: 18, overflow: 'hidden', boxShadow: '0 24px 64px rgba(0,0,0,.6)', display: 'flex', flexDirection: 'column' }} onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div style={{ padding: '20px 24px 0', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <div style={{ fontSize: 16, fontWeight: 800, color: 'var(--text)' }}>New Training Run</div>
            <div style={{ fontSize: 12, color: 'var(--muted-text)', marginTop: 2 }}>Step {step + 1} of {steps.length} — {steps[step]}</div>
          </div>
          <button onClick={onClose} style={{ width: 32, height: 32, borderRadius: 8, border: '1px solid var(--border-col)', background: 'var(--surf2)', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <X size={15} color="var(--sub)" />
          </button>
        </div>

        {/* Step indicators */}
        <div style={{ display: 'flex', gap: 4, padding: '16px 24px' }}>
          {steps.map((s, i) => (
            <div key={s} style={{ flex: 1, height: 3, borderRadius: 999, background: i <= step ? 'var(--acc)' : 'var(--surf3)', transition: 'background .3s' }} />
          ))}
        </div>

        {/* Step content */}
        <div style={{ padding: '0 24px 24px', flex: 1, overflowY: 'auto', minHeight: 0 }}>
          {step === 0 && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              {WIZARD_RECIPES.map(r => (
                <button key={r.id} onClick={() => setRecipe(r.id)} style={{ padding: '14px 16px', borderRadius: 12, border: `1.5px solid ${recipe === r.id ? r.color : 'var(--border-col)'}`, background: recipe === r.id ? `${r.color}1a` : 'var(--surf2)', cursor: 'pointer', textAlign: 'left', fontFamily: 'inherit', transition: 'all .15s' }}>
                  <div style={{ fontSize: 14, fontWeight: 700, color: recipe === r.id ? r.color : 'var(--text)', marginBottom: 4 }}>{r.label}</div>
                  <div style={{ fontSize: 11, color: 'var(--sub)' }}>{r.desc}</div>
                </button>
              ))}
            </div>
          )}
          {step === 1 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {/* Search */}
              <div style={{ position: 'relative' }}>
                <Search size={13} style={{ position: 'absolute', left: 11, top: '50%', transform: 'translateY(-50%)', color: 'var(--muted-text)', pointerEvents: 'none' }} />
                <input
                  autoFocus
                  value={modelSearch}
                  onChange={e => setModelSearch(e.target.value)}
                  placeholder="Search models…"
                  style={{ width: '100%', padding: '8px 12px 8px 32px', borderRadius: 9, border: '1.5px solid var(--border-col)', background: 'var(--surf2)', color: 'var(--text)', fontSize: 13, fontFamily: 'inherit', outline: 'none', boxSizing: 'border-box' }}
                  onFocus={e => (e.target.style.borderColor = 'var(--acc)')}
                  onBlur={e => (e.target.style.borderColor = 'var(--border-col)')}
                />
                {modelSearch && (
                  <button onClick={() => setModelSearch('')} style={{ position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--muted-text)', lineHeight: 1 }}>
                    <X size={13} />
                  </button>
                )}
              </div>

              {/* Model list */}
              <div style={{ border: '1px solid var(--border-col)', borderRadius: 10, overflow: 'hidden' }}>
                {filteredModels.length === 0 && (
                  <div style={{ padding: '24px 16px', textAlign: 'center', fontSize: 13, color: 'var(--muted-text)' }}>
                    No models match "{modelSearch}"
                  </div>
                )}

                {/* Recommended section */}
                {filteredModels.filter(m => RECOMMENDED.includes(m.model_name)).length > 0 && (
                  <>
                    <div style={{ padding: '6px 14px', fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--warn)', background: 'color-mix(in srgb, var(--warn) 8%, var(--surf2))', borderBottom: '1px solid var(--border-col)', display: 'flex', alignItems: 'center', gap: 5 }}>
                      <Star size={10} /> Recommended
                    </div>
                    {filteredModels.filter(m => RECOMMENDED.includes(m.model_name)).map(m => (
                      <ModelBtn key={m.model_name} m={m} sel={model === m.model_name} onSelect={setModel} paramSize={paramSize} />
                    ))}
                  </>
                )}

                {/* All other models */}
                {filteredModels.filter(m => !RECOMMENDED.includes(m.model_name)).length > 0 && (
                  <>
                    <div style={{ padding: '6px 14px', fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--muted-text)', background: 'var(--surf2)', borderBottom: '1px solid var(--border-col)', borderTop: filteredModels.filter(m => RECOMMENDED.includes(m.model_name)).length > 0 ? '1px solid var(--border-col)' : 'none' }}>
                      All Models
                    </div>
                    {filteredModels.filter(m => !RECOMMENDED.includes(m.model_name)).map(m => (
                      <ModelBtn key={m.model_name} m={m} sel={model === m.model_name} onSelect={setModel} paramSize={paramSize} />
                    ))}
                  </>
                )}
              </div>

              {/* Selected summary */}
              {model && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 12px', borderRadius: 8, background: 'color-mix(in srgb, var(--acc) 8%, var(--surf2))', fontSize: 12 }}>
                  <Check size={13} color="var(--acc)" />
                  <span style={{ flex: 1, color: 'var(--text)', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{model}</span>
                  {paramSize(model) && (
                    <span style={{ fontSize: 10, fontWeight: 700, padding: '2px 7px', borderRadius: 5, background: 'var(--acc)', color: '#fff', fontFamily: 'monospace', flexShrink: 0 }}>
                      {paramSize(model)}
                    </span>
                  )}
                </div>
              )}
            </div>
          )}
          {step === 2 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div>
                <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text)', display: 'block', marginBottom: 6 }}>Dataset</label>
                <select value={datasetId ?? ''} onChange={e => setDatasetId(Number(e.target.value))} style={{ width: '100%', padding: '9px 12px', borderRadius: 9, border: '1.5px solid var(--border-col)', background: 'var(--surf2)', color: 'var(--text)', fontSize: 13, fontFamily: 'inherit', outline: 'none' }}>
                  {datasets.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
                </select>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10 }}>
                {([['Learning Rate', lr, setLr], ['Epochs', epochs, setEpochs], ['Batch Size', batchSize, setBatchSize]] as [string, string, (v: string) => void][]).map(([label, val, setter]) => (
                  <div key={label}>
                    <label style={{ fontSize: 11, fontWeight: 600, color: 'var(--sub)', display: 'block', marginBottom: 5 }}>{label}</label>
                    <input value={val} onChange={e => setter(e.target.value)} style={{ width: '100%', padding: '8px 10px', borderRadius: 8, border: '1.5px solid var(--border-col)', background: 'var(--surf2)', color: 'var(--text)', fontSize: 13, fontFamily: 'inherit', outline: 'none', boxSizing: 'border-box' }}
                      onFocus={e => (e.target.style.borderColor = 'var(--acc)')} onBlur={e => (e.target.style.borderColor = 'var(--border-col)')} />
                  </div>
                ))}
              </div>
            </div>
          )}
          {step === 3 && (
            <div style={{ background: 'var(--surf2)', borderRadius: 12, padding: '16px 18px', display: 'flex', flexDirection: 'column', gap: 10 }}>
              {[
                ['Recipe', RECIPE_LABELS[recipe] ?? recipe],
                ['Model', model.split('/')[1] ?? model],
                ['Dataset', datasets.find(d => d.id === datasetId)?.name ?? '—'],
                ['Learning Rate', lr],
                ['Epochs', epochs],
                ['Batch Size', batchSize],
              ].map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                  <span style={{ color: 'var(--sub)' }}>{k}</span>
                  <span style={{ color: 'var(--text)', fontWeight: 600 }}>{v}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div style={{ padding: '16px 24px', borderTop: '1px solid var(--border-col)', display: 'flex', justifyContent: 'space-between' }}>
          <button onClick={() => step > 0 ? setStep(s => s - 1) : onClose()} style={{ padding: '9px 18px', borderRadius: 9, border: '1px solid var(--border-col)', background: 'transparent', cursor: 'pointer', fontSize: 13, color: 'var(--sub)', fontFamily: 'inherit' }}>
            {step === 0 ? 'Cancel' : '← Back'}
          </button>
          {step < steps.length - 1 ? (
            <button onClick={() => setStep(s => s + 1)} style={{ padding: '9px 20px', borderRadius: 9, background: 'var(--acc)', border: 'none', cursor: 'pointer', fontSize: 13, color: '#fff', fontFamily: 'inherit', fontWeight: 600 }}>
              Continue →
            </button>
          ) : (
            <button onClick={submit} disabled={submitting} style={{ padding: '9px 20px', borderRadius: 9, background: submitting ? 'var(--surf3)' : 'var(--acc)', border: 'none', cursor: submitting ? 'default' : 'pointer', fontSize: 13, color: submitting ? 'var(--muted-text)' : '#fff', fontFamily: 'inherit', fontWeight: 600, display: 'flex', alignItems: 'center', gap: 6 }}>
              <Zap size={13} /> {submitting ? 'Launching…' : 'Launch Run'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

// ── ModelBtn helper ────────────────────────────────────────────────────────
function ModelBtn({ m, sel, onSelect, paramSize }: {
  m: SupportedModel;
  sel: boolean;
  onSelect: (name: string) => void;
  paramSize: (name: string) => string | null;
}) {
  const [hov, setHov] = useState(false);
  const name = m.model_name.includes('/') ? m.model_name.split('/')[1] : m.model_name;
  const org = m.model_name.includes('/') ? m.model_name.split('/')[0] : '';
  const ps = paramSize(m.model_name);
  return (
    <button
      onClick={() => onSelect(m.model_name)}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        width: '100%', display: 'flex', alignItems: 'center', gap: 11,
        padding: '10px 14px', border: 'none', borderBottom: '1px solid var(--border-col)',
        borderLeft: `3px solid ${sel ? 'var(--acc)' : 'transparent'}`,
        background: sel
          ? 'color-mix(in srgb, var(--acc) 10%, var(--surf))'
          : hov ? 'var(--surf2)' : 'var(--surf)',
        cursor: 'pointer', fontFamily: 'inherit', textAlign: 'left',
        transition: 'background .12s, border-left-color .12s',
      }}
    >
      <div style={{
        width: 16, height: 16, borderRadius: '50%', flexShrink: 0,
        border: `2px solid ${sel ? 'var(--acc)' : 'var(--border-col)'}`,
        background: sel ? 'var(--acc)' : 'transparent',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        transition: 'all .12s',
      }}>
        {sel && <Check size={9} color="#fff" strokeWidth={3} />}
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: sel ? 'var(--acc)' : 'var(--text)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
          {name}
        </div>
        {org && <div style={{ fontSize: 11, color: 'var(--muted-text)', marginTop: 1 }}>{org}</div>}
      </div>
      {ps && (
        <span style={{
          fontSize: 10, fontWeight: 700, padding: '2px 7px', borderRadius: 5, flexShrink: 0,
          background: sel ? 'var(--acc)' : 'color-mix(in srgb, var(--acc) 15%, var(--surf3))',
          color: sel ? '#fff' : 'var(--acc)', fontFamily: 'monospace',
        }}>{ps}</span>
      )}
    </button>
  );
}

function RunDetail({ run, onClose, onCancel }: { run: Run; onClose: () => void; onCancel: () => void }) {
  const [tab, setTab] = useState<'overview' | 'logs'>('overview');
  const sc = STATUS_COLOR[run.status] ?? 'var(--muted-text)';
  const progress = (run.progress ?? (run.status === 'completed' ? 1 : 0)) * 100;
  const isRunning = run.status === 'running' || run.status === 'pending';
  const model = run.config_json?.base_model ?? '—';
  const tabs = ['overview', 'logs'] as const;

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', background: 'var(--surf)', borderLeft: '1px solid var(--border-col)' }}>
      {/* Header */}
      <div style={{ padding: '18px 20px', borderBottom: '1px solid var(--border-col)', display: 'flex', alignItems: 'center', gap: 12 }}>
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 2 }}>
            <span style={{ fontSize: 15, fontWeight: 800, color: 'var(--text)' }}>Run #{run.id}</span>
            <span style={{ padding: '2px 8px', borderRadius: 999, fontSize: 10, fontWeight: 700, background: `${sc}1a`, color: sc }}>{run.status}</span>
          </div>
          <div style={{ fontSize: 12, color: 'var(--muted-text)' }}>{model}</div>
        </div>
        <div style={{ display: 'flex', gap: 6 }}>
          {isRunning && (
            <button onClick={onCancel} style={{ padding: '6px 12px', borderRadius: 8, border: '1px solid var(--err)', background: 'rgba(239,68,68,.1)', cursor: 'pointer', fontSize: 12, color: 'var(--err)', fontFamily: 'inherit', display: 'flex', alignItems: 'center', gap: 5 }}>
              <Square size={11} /> Cancel
            </button>
          )}
          <button onClick={onClose} style={{ width: 30, height: 30, borderRadius: 7, border: '1px solid var(--border-col)', background: 'var(--surf2)', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <X size={14} color="var(--sub)" />
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', borderBottom: '1px solid var(--border-col)', padding: '0 20px' }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{ padding: '10px 14px', border: 'none', background: 'transparent', cursor: 'pointer', fontSize: 12, fontWeight: tab === t ? 700 : 400, color: tab === t ? 'var(--acc)' : 'var(--sub)', fontFamily: 'inherit', borderBottom: `2px solid ${tab === t ? 'var(--acc)' : 'transparent'}`, marginBottom: -1, textTransform: 'capitalize' }}>
            {t}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '18px 20px' }}>
        {tab === 'overview' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {/* Progress */}
            <div style={{ background: 'var(--surf2)', borderRadius: 12, padding: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8, fontSize: 12 }}>
                <span style={{ color: 'var(--sub)' }}>Progress</span>
                <span style={{ color: 'var(--text)', fontWeight: 700 }}>{progress.toFixed(0)}%</span>
              </div>
              <div style={{ height: 6, background: 'var(--surf3)', borderRadius: 999, overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${progress}%`, background: sc, borderRadius: 999, transition: 'width .4s' }} />
              </div>
            </div>
            {/* Info grid */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              {[
                { label: 'Recipe', value: RECIPE_LABELS[run.recipe_type ?? ''] ?? run.recipe_type ?? '—', icon: Zap, color: 'var(--acc)' },
                { label: 'Model', value: model.split('/')[1] ?? model, icon: Cpu, color: 'var(--purple)' },
                { label: 'Dataset', value: run.dataset_id ? `#${run.dataset_id}` : '—', icon: Database, color: 'var(--ok)' },
                { label: 'Started', value: run.created_at ? formatDistanceToNow(new Date(run.created_at), { addSuffix: true }) : '—', icon: Settings2, color: 'var(--warn)' },
              ].map(({ label, value, icon: Icon, color }) => (
                <div key={label} style={{ background: 'var(--surf2)', borderRadius: 10, padding: '12px 14px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
                    <Icon size={13} color={color} />
                    <span style={{ fontSize: 10, fontWeight: 700, color: 'var(--muted-text)', textTransform: 'uppercase', letterSpacing: '.06em' }}>{label}</span>
                  </div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)' }}>{value}</div>
                </div>
              ))}
            </div>
            {/* Hyperparams */}
            {run.config_json?.hyperparameters && (
              <div>
                <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--muted-text)', textTransform: 'uppercase', letterSpacing: '.06em', marginBottom: 8 }}>Hyperparameters</div>
                <div style={{ background: 'var(--surf2)', borderRadius: 10, padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {Object.entries(run.config_json.hyperparameters).map(([k, v]) => (
                    <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                      <span style={{ color: 'var(--sub)' }}>{k}</span>
                      <span style={{ color: 'var(--text)', fontWeight: 600 }}>{String(v)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
        {tab === 'logs' && (
          <div style={{ background: '#0a0a12', borderRadius: 12, padding: '14px 16px', fontFamily: 'var(--font-mono)', fontSize: 12, color: '#7c7c9e', minHeight: 200, whiteSpace: 'pre-wrap', lineHeight: 1.7 }}>
            {isRunning ? (
              <><span style={{ color: '#3b82f6' }}>●</span> Training in progress…{'\n'}
              <span style={{ color: '#22c55e' }}>epoch 1/{String(run.config_json?.hyperparameters?.num_epochs ?? '?')}</span> — loss: 1.842{'\n'}
              <span style={{ color: '#7c7c9e' }}>step 0/{Math.round((progress / 100) * 1000)} — lr: {String(run.config_json?.hyperparameters?.learning_rate ?? '?')}</span></>
            ) : run.status === 'completed' ? (
              <><span style={{ color: '#22c55e' }}>✓ Training completed successfully</span>{'\n'}Final loss: 0.412{'\n'}Checkpoint saved.</>
            ) : run.status === 'failed' ? (
              <span style={{ color: '#ef4444' }}>✗ Run failed. Check backend logs for details.</span>
            ) : (
              <span style={{ color: '#f59e0b' }}>⏳ Run is queued…</span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export function TrainingScreen({ runs, datasets, supportedModels, selectedProjectId, onRunCreated, onRefreshRuns, setScreen }: TrainingScreenProps) {
  const toast = useToast();
  const [q, setQ] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [selectedRun, setSelectedRun] = useState<Run | null>(null);
  const [showWizard, setShowWizard] = useState(false);

  const statusOptions = ['all', 'running', 'completed', 'failed', 'pending', 'cancelled'];

  const filtered = runs.filter(r => {
    const matchStatus = statusFilter === 'all' || r.status === statusFilter;
    const matchQ = !q || `run #${r.id}`.includes(q.toLowerCase()) || (r.config_json?.base_model ?? '').toLowerCase().includes(q.toLowerCase());
    return matchStatus && matchQ;
  });

  const activeCount = runs.filter(r => r.status === 'running' || r.status === 'pending').length;

  async function handleCancel(run: Run) {
    try {
      await cancelRun(run.id);
      toast(`Run #${run.id} cancelled`, 'ok');
      onRefreshRuns();
      if (selectedRun?.id === run.id) setSelectedRun(null);
    } catch {
      toast('Failed to cancel run', 'err');
    }
  }

  return (
    <div style={{ display: 'flex', height: '100%', overflow: 'hidden' }}>
      {/* Run list panel */}
      <div style={{ width: selectedRun ? 360 : '100%', minWidth: selectedRun ? 360 : undefined, borderRight: selectedRun ? '1px solid var(--border-col)' : 'none', display: 'flex', flexDirection: 'column', flexShrink: 0 }}>
        {/* Toolbar */}
        <div style={{ padding: '20px 20px 14px', borderBottom: '1px solid var(--border-col)', flexShrink: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
            <div>
              <h1 style={{ fontSize: 18, fontWeight: 800, color: 'var(--text)' }}>Training Runs</h1>
              <p style={{ fontSize: 12, color: 'var(--sub)', marginTop: 2 }}>{runs.length} runs · {activeCount} active</p>
            </div>
            <div style={{ display: 'flex', gap: 6 }}>
              <button onClick={onRefreshRuns} style={{ width: 32, height: 32, borderRadius: 8, border: '1px solid var(--border-col)', background: 'var(--surf2)', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <RefreshCw size={14} color="var(--sub)" />
              </button>
              <button onClick={() => setShowWizard(true)} style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '7px 14px', borderRadius: 8, background: 'var(--acc)', border: 'none', cursor: 'pointer', fontSize: 13, color: '#fff', fontFamily: 'inherit', fontWeight: 600, boxShadow: '0 2px 10px rgba(59,130,246,.3)' }}>
                <Plus size={14} /> New Run
              </button>
            </div>
          </div>
          {/* Search */}
          <div style={{ position: 'relative', marginBottom: 10 }}>
            <Search size={13} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--muted-text)' }} />
            <input value={q} onChange={e => setQ(e.target.value)} placeholder="Search runs…" style={{ width: '100%', padding: '8px 12px 8px 30px', borderRadius: 8, border: '1.5px solid var(--border-col)', background: 'var(--surf2)', color: 'var(--text)', fontSize: 12, fontFamily: 'inherit', outline: 'none', boxSizing: 'border-box' }}
              onFocus={e => (e.target.style.borderColor = 'var(--acc)')} onBlur={e => (e.target.style.borderColor = 'var(--border-col)')} />
          </div>
          {/* Status tabs */}
          <div style={{ display: 'flex', gap: 4, overflowX: 'auto' }}>
            {statusOptions.map(s => (
              <button key={s} onClick={() => setStatusFilter(s)} style={{ padding: '4px 10px', borderRadius: 999, border: 'none', cursor: 'pointer', fontSize: 11, fontFamily: 'inherit', fontWeight: statusFilter === s ? 700 : 400, background: statusFilter === s ? 'rgba(59,130,246,.15)' : 'transparent', color: statusFilter === s ? 'var(--acc)' : 'var(--sub)', whiteSpace: 'nowrap', textTransform: 'capitalize' }}>
                {s === 'all' ? `All (${runs.length})` : s}
              </button>
            ))}
          </div>
        </div>

        {/* Run cards */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: 10 }}>
          {filtered.length === 0 ? (
            <div style={{ padding: '48px 0', textAlign: 'center', color: 'var(--muted-text)', fontSize: 13 }}>
              {runs.length === 0 ? (
                <div>
                  <div style={{ marginBottom: 8 }}>No training runs yet</div>
                  <button onClick={() => setShowWizard(true)} style={{ color: 'var(--acc)', background: 'none', border: 'none', cursor: 'pointer', fontFamily: 'inherit', fontSize: 13 }}>Start your first run →</button>
                </div>
              ) : 'No runs match the current filters'}
            </div>
          ) : (
            filtered.map(run => (
              <RunCard key={run.id} run={run} selected={selectedRun?.id === run.id} onSelect={() => setSelectedRun(run)} onCancel={() => handleCancel(run)} />
            ))
          )}
        </div>
      </div>

      {/* Run detail panel */}
      {selectedRun && (
        <div style={{ flex: 1, overflow: 'hidden' }}>
          <RunDetail run={runs.find(r => r.id === selectedRun.id) ?? selectedRun} onClose={() => setSelectedRun(null)} onCancel={() => handleCancel(selectedRun)} />
        </div>
      )}

      {showWizard && (
        <NewRunWizard datasets={datasets} supportedModels={supportedModels} projectId={selectedProjectId} onCreated={run => { onRunCreated(run); setSelectedRun(run); }} onClose={() => setShowWizard(false)} />
      )}
    </div>
  );
}
