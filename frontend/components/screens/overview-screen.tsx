'use client';

import { Sparkline } from '@/components/ui/sparkline';
import { useToast } from '@/components/ui/toast-provider';
import { Check, FileText, Database, Zap, AlertTriangle, ChevronRight } from 'lucide-react';
import type { Project, Run, Dataset } from '@/lib/api';
import { formatDistanceToNow } from 'date-fns';

type Screen = 'overview' | 'training' | 'models' | 'chat' | 'datasets' | 'deployments';

interface OverviewScreenProps {
  projects: Project[];
  runs: Run[];
  datasets: Dataset[];
  setScreen: (s: Screen) => void;
}

const PIPELINE = [
  { id: 1, label: 'Choose Model', screen: 'models' as Screen },
  { id: 2, label: 'Upload Data',  screen: 'datasets' as Screen },
  { id: 3, label: 'Train',        screen: 'training' as Screen },
  { id: 4, label: 'Evaluate',     screen: 'chat' as Screen },
  { id: 5, label: 'Deploy',       screen: 'deployments' as Screen },
];

const STATUS_COLOR: Record<string, string> = {
  completed: 'var(--ok)',
  running: 'var(--blue)',
  failed: 'var(--err)',
  pending: 'var(--warn)',
};

function PulseDot({ color }: { color: string }) {
  return (
    <div style={{ position: 'relative', width: 8, height: 8, flexShrink: 0 }}>
      <div className="animate-pulse-dot" style={{ position: 'absolute', inset: -3, borderRadius: 999, background: color, opacity: .35 }} />
      <div style={{ width: 8, height: 8, borderRadius: 999, background: color }} />
    </div>
  );
}

function ProgressBar({ value, color, animated }: { value: number; color: string; animated?: boolean }) {
  return (
    <div style={{ height: 5, background: 'var(--surf3)', borderRadius: 999, overflow: 'hidden' }}>
      <div style={{ height: '100%', width: `${value}%`, background: color, borderRadius: 999, transition: 'width .4s ease', position: 'relative', overflow: 'hidden' }}>
        {animated && value < 100 && (
          <div className="animate-shimmer" style={{ position: 'absolute', inset: 0, background: 'linear-gradient(90deg,transparent,rgba(255,255,255,.3),transparent)', backgroundSize: '200% 100%' }} />
        )}
      </div>
    </div>
  );
}

export function OverviewScreen({ projects, runs, datasets, setScreen }: OverviewScreenProps) {
  const toast = useToast();
  const activeRuns = runs.filter(r => r.status === 'running' || r.status === 'pending');
  const completedRuns = runs.filter(r => r.status === 'completed');

  // Determine pipeline step
  const pipelineStep: number = datasets.length === 0 ? 1 : runs.length === 0 ? 2 : activeRuns.length > 0 ? 3 : 4;

  const stats = [
    { label: 'Projects',    value: projects.length, icon: FileText,  color: 'var(--acc)',    spk: [1,2,3,4,5,6,7,projects.length] },
    { label: 'Datasets',    value: datasets.length, icon: Database,  color: 'var(--purple)', spk: [0,1,1,2,2,3,datasets.length,datasets.length] },
    { label: 'Total Runs',  value: runs.length,     icon: Zap,       color: 'var(--ok)',     spk: [0,1,2,3,4,5,runs.length > 2 ? runs.length-2 : 0, runs.length], trend: runs.length > 0 ? `↑ ${Math.min(3, runs.length)} this week` : undefined },
    { label: 'Active Runs', value: activeRuns.length, icon: AlertTriangle, color: 'var(--warn)', spk: [0,1,0,1,0,activeRuns.length,activeRuns.length,activeRuns.length] },
  ];

  const recentRuns = [...runs].slice(0, 5);

  return (
    <div style={{ padding: '28px 32px', overflowY: 'auto', height: '100%' }} className="animate-fadeUp">
      {/* Pipeline */}
      <div style={{ background: 'var(--surf)', border: '1px solid var(--border-col)', borderRadius: 16, padding: '22px 28px', marginBottom: 22, boxShadow: 'var(--shadow-c)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 18 }}>
          <div>
            <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--text)' }}>Fine-tuning Pipeline</div>
            <div style={{ fontSize: 12, color: 'var(--sub)', marginTop: 2 }}>
              {pipelineStep === 2 ? "Step 1 done — upload your training data next" :
               pipelineStep === 3 ? "Ready to launch training" :
               activeRuns.length > 0 ? `Training in progress — ${activeRuns.length} active run${activeRuns.length > 1 ? 's' : ''}` :
               "Pipeline complete — evaluate and deploy"}
            </div>
          </div>
          <button
            onClick={() => setScreen('training')}
            style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '8px 16px', borderRadius: 9, background: 'var(--acc)', color: '#fff', border: 'none', cursor: 'pointer', fontFamily: 'inherit', fontSize: 13, fontWeight: 500, boxShadow: '0 2px 10px rgba(59,130,246,.3)' }}
            onMouseEnter={e => (e.currentTarget.style.background = 'var(--acc-hover)')}
            onMouseLeave={e => (e.currentTarget.style.background = 'var(--acc)')}
          >
            <Zap size={14} color="#fff" /> Start Training
          </button>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 0 }}>
          {PIPELINE.map((step, i) => {
            const done = i + 1 < pipelineStep;
            const active = i + 1 === pipelineStep;
            return (
              <div key={step.id} style={{ display: 'flex', alignItems: 'center', flex: i < PIPELINE.length - 1 ? undefined : undefined }}>
                <div
                  onClick={() => setScreen(step.screen)}
                  style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', cursor: 'pointer', minWidth: 90 }}
                >
                  <div style={{
                    width: 42, height: 42, borderRadius: 12,
                    background: done ? 'rgba(59,130,246,.1)' : active ? 'rgba(59,130,246,.08)' : 'var(--surf2)',
                    border: `2px solid ${done || active ? 'var(--acc)' : 'var(--border-col)'}`,
                    display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 8,
                    boxShadow: active ? '0 0 0 4px rgba(59,130,246,.2)' : 'none', transition: 'all .2s',
                  }}>
                    {done ? <Check size={18} color="var(--acc)" /> : <Zap size={18} color={active ? 'var(--acc)' : 'var(--muted-text)'} />}
                  </div>
                  <div style={{ fontSize: 11, fontWeight: active ? 700 : 500, color: active ? 'var(--acc)' : done ? 'var(--text)' : 'var(--muted-text)', textAlign: 'center' }}>{step.label}</div>
                  {active && <div style={{ width: 5, height: 5, borderRadius: 999, background: 'var(--acc)', marginTop: 4 }} />}
                </div>
                {i < PIPELINE.length - 1 && (
                  <div style={{ flex: 1, height: 2, background: done ? 'var(--acc)' : 'var(--border-col)', margin: '0 4px 24px', minWidth: 24, transition: 'background .3s' }} />
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Stats */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 14, marginBottom: 22 }}>
        {stats.map(({ label, value, icon: Icon, color, spk, trend }) => (
          <div key={label} className="hover-lift" style={{ background: 'var(--surf)', border: '1px solid var(--border-col)', borderRadius: 14, padding: '18px 20px', boxShadow: 'var(--shadow-c)', cursor: 'default' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
              <div>
                <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--muted-text)', textTransform: 'uppercase', letterSpacing: '.07em', marginBottom: 6 }}>{label}</div>
                <div style={{ fontSize: 30, fontWeight: 800, color: 'var(--text)', lineHeight: 1 }}>{value}</div>
                {trend && <div style={{ fontSize: 11, color: 'var(--ok)', marginTop: 5, fontWeight: 500 }}>{trend}</div>}
              </div>
              <div style={{ width: 36, height: 36, borderRadius: 10, background: `${color}1a`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Icon size={16} color={color} />
              </div>
            </div>
            <Sparkline data={spk} color={color} width={120} height={26} />
          </div>
        ))}
      </div>

      {/* Bottom row */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: 16 }}>
        {/* Recent runs */}
        <div style={{ background: 'var(--surf)', border: '1px solid var(--border-col)', borderRadius: 14, overflow: 'hidden', boxShadow: 'var(--shadow-c)' }}>
          <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border-col)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--text)' }}>Recent Runs</div>
            <button onClick={() => setScreen('training')} style={{ fontSize: 12, color: 'var(--acc)', background: 'none', border: 'none', cursor: 'pointer', fontFamily: 'inherit', fontWeight: 500 }}>View all →</button>
          </div>
          {recentRuns.length === 0 ? (
            <div style={{ padding: '32px', textAlign: 'center', color: 'var(--muted-text)', fontSize: 13 }}>No runs yet. <button onClick={() => setScreen('training')} style={{ color: 'var(--acc)', background: 'none', border: 'none', cursor: 'pointer', fontFamily: 'inherit', fontSize: 13 }}>Start training →</button></div>
          ) : (
            recentRuns.map(run => {
              const sc = STATUS_COLOR[run.status] ?? 'var(--muted-text)';
              const isRunning = run.status === 'running';
              return (
                <div key={run.id} style={{ display: 'flex', alignItems: 'center', gap: 14, padding: '14px 20px', borderBottom: '1px solid var(--border-light)', cursor: 'pointer', transition: 'background .12s' }}
                  onMouseEnter={e => (e.currentTarget.style.background = 'var(--surf2)')}
                  onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
                >
                  {isRunning ? <PulseDot color={sc} /> : <div style={{ width: 8, height: 8, borderRadius: 999, background: sc, flexShrink: 0 }} />}
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)' }}>Run #{run.id}</div>
                    <div style={{ fontSize: 11, color: 'var(--sub)' }}>{run.config_json?.base_model ?? '—'} · {run.recipe_type}</div>
                  </div>
                  <div style={{ width: 90 }}>
                    <ProgressBar value={(run.progress ?? (run.status === 'completed' ? 1 : 0)) * 100} color={sc} animated={isRunning} />
                    <div style={{ fontSize: 10, color: 'var(--muted-text)', marginTop: 2, textAlign: 'right' }}>
                      {((run.progress ?? (run.status === 'completed' ? 1 : 0)) * 100).toFixed(0)}%
                    </div>
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--muted-text)', minWidth: 60, textAlign: 'right' }}>
                    {run.created_at ? formatDistanceToNow(new Date(run.created_at), { addSuffix: true }) : '—'}
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Activity feed */}
        <div style={{ background: 'var(--surf)', border: '1px solid var(--border-col)', borderRadius: 14, overflow: 'hidden', boxShadow: 'var(--shadow-c)' }}>
          <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border-col)' }}>
            <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--text)' }}>Activity</div>
          </div>
          <div style={{ padding: '12px 16px', display: 'flex', flexDirection: 'column', gap: 0 }}>
            {runs.slice(0, 5).map((run, i) => {
              const color = STATUS_COLOR[run.status] ?? 'var(--muted-text)';
              const msg = run.status === 'completed' ? `Run #${run.id} completed — ${run.recipe_type}`
                        : run.status === 'running'   ? `Run #${run.id} training — ${run.recipe_type}`
                        : run.status === 'failed'    ? `Run #${run.id} failed`
                        : `Run #${run.id} pending`;
              return (
                <div key={run.id} style={{ display: 'flex', gap: 10, padding: '9px 0', borderBottom: i < 4 ? '1px solid var(--border-light)' : 'none' }}>
                  <div style={{ width: 26, height: 26, borderRadius: 7, background: `${color}1a`, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                    <Zap size={12} color={color} />
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 12, color: 'var(--text)', lineHeight: 1.5 }}>{msg}</div>
                    <div style={{ fontSize: 10, color: 'var(--muted-text)', marginTop: 1 }}>
                      {run.created_at ? formatDistanceToNow(new Date(run.created_at), { addSuffix: true }) : '—'}
                    </div>
                  </div>
                </div>
              );
            })}
            {runs.length === 0 && (
              <div style={{ padding: '24px 0', textAlign: 'center', color: 'var(--muted-text)', fontSize: 12 }}>No activity yet</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
