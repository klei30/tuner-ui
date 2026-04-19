'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { Rocket, Copy, ExternalLink, Globe, Lock, RefreshCw, Code2, CheckCircle2, XCircle, Clock, Loader2 } from 'lucide-react';
import { useToast } from '@/components/ui/toast-provider';
import { formatDistanceToNow } from 'date-fns';

type Screen = 'overview' | 'training' | 'models' | 'chat' | 'datasets' | 'deployments';

interface Deployment {
  id: number;
  checkpoint_id: number;
  hf_repo_name: string;
  hf_repo_url: string;
  hf_model_id: string;
  is_private: boolean;
  merged_weights: boolean;
  status: string;
  deployed_at: string | null;
  error_message: string | null;
}

interface DeploymentsScreenProps {
  setScreen: (s: Screen) => void;
}

const STATUS_CONFIG: Record<string, { color: string; icon: React.ElementType; label: string }> = {
  completed: { color: 'var(--ok)', icon: CheckCircle2, label: 'Live' },
  running:   { color: 'var(--blue)', icon: Loader2, label: 'Deploying' },
  failed:    { color: 'var(--err)', icon: XCircle, label: 'Failed' },
  pending:   { color: 'var(--warn)', icon: Clock, label: 'Queued' },
};

const CODE_SNIPPETS: Record<string, (repoId: string) => string> = {
  Python: (id) => `from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "${id}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

inputs = tokenizer("Hello!", return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(output[0]))`,
  cURL: (id) => `curl https://api-inference.huggingface.co/models/${id} \\
  -H "Authorization: Bearer $HF_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{"inputs": "Hello!"}'`,
  JavaScript: (id) => `import { HfInference } from "@huggingface/inference";

const hf = new HfInference(process.env.HF_TOKEN);
const result = await hf.textGeneration({
  model: "${id}",
  inputs: "Hello!",
  parameters: { max_new_tokens: 128 },
});
console.log(result.generated_text);`,
};

function DeploymentCard({ dep, onTestInChat }: { dep: Deployment; onTestInChat: () => void }) {
  const toast = useToast();
  const [codeTab, setCodeTab] = useState<'Python' | 'cURL' | 'JavaScript'>('Python');
  const [showCode, setShowCode] = useState(false);
  const cfg = STATUS_CONFIG[dep.status] ?? STATUS_CONFIG.pending;
  const StatusIcon = cfg.icon;
  const isActive = dep.status === 'completed';

  return (
    <div className="hover-lift" style={{ background: 'var(--surf)', border: '1px solid var(--border-col)', borderRadius: 16, overflow: 'hidden', boxShadow: 'var(--shadow-c)', position: 'relative' }}>
      {/* Animated top bar */}
      <div style={{ height: 3, background: cfg.color, position: 'relative', overflow: 'hidden' }}>
        {dep.status === 'running' && <div className="animate-shimmer" style={{ position: 'absolute', inset: 0, background: 'linear-gradient(90deg,transparent,rgba(255,255,255,.4),transparent)', backgroundSize: '200% 100%' }} />}
      </div>

      <div style={{ padding: '18px 20px' }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 14, marginBottom: 14 }}>
          <div style={{ width: 42, height: 42, borderRadius: 12, background: `${cfg.color}1a`, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
            <Rocket size={19} color={cfg.color} />
          </div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 3, flexWrap: 'wrap' }}>
              <span style={{ fontSize: 14, fontWeight: 700, color: 'var(--text)' }}>{dep.hf_repo_name}</span>
              <span style={{ display: 'flex', alignItems: 'center', gap: 4, padding: '2px 8px', borderRadius: 999, fontSize: 10, fontWeight: 700, background: `${cfg.color}1a`, color: cfg.color }}>
                <StatusIcon size={10} /> {cfg.label}
              </span>
              {dep.is_private ? <span style={{ display: 'flex', alignItems: 'center', gap: 4, padding: '2px 7px', borderRadius: 999, fontSize: 10, background: 'var(--surf2)', color: 'var(--muted-text)' }}><Lock size={9} /> Private</span>
                : <span style={{ display: 'flex', alignItems: 'center', gap: 4, padding: '2px 7px', borderRadius: 999, fontSize: 10, background: 'rgba(34,197,94,.1)', color: 'var(--ok)' }}><Globe size={9} /> Public</span>}
            </div>
            <div style={{ fontSize: 12, color: 'var(--muted-text)', fontFamily: 'var(--font-mono)' }}>{dep.hf_model_id}</div>
          </div>
        </div>

        {/* Stats row */}
        <div style={{ display: 'flex', gap: 14, marginBottom: 14, fontSize: 12 }}>
          <span style={{ color: 'var(--muted-text)' }}>Checkpoint #{dep.checkpoint_id}</span>
          {dep.deployed_at && <span style={{ color: 'var(--muted-text)' }}>{formatDistanceToNow(new Date(dep.deployed_at), { addSuffix: true })}</span>}
          {dep.merged_weights && <span style={{ color: 'var(--ok)', fontWeight: 600 }}>✓ Merged weights</span>}
        </div>

        {dep.error_message && (
          <div style={{ padding: '8px 12px', borderRadius: 8, background: 'rgba(239,68,68,.08)', border: '1px solid rgba(239,68,68,.2)', fontSize: 12, color: 'var(--err)', marginBottom: 14 }}>
            {dep.error_message}
          </div>
        )}

        {/* Endpoint URL */}
        {isActive && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 12px', borderRadius: 9, background: 'var(--surf2)', border: '1px solid var(--border-col)', marginBottom: 14 }}>
            <code style={{ flex: 1, fontSize: 11, color: 'var(--sub)', fontFamily: 'var(--font-mono)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {dep.hf_repo_url}
            </code>
            <button onClick={() => { navigator.clipboard.writeText(dep.hf_repo_url); toast('URL copied', 'ok'); }} style={{ width: 26, height: 26, borderRadius: 6, border: '1px solid var(--border-col)', background: 'transparent', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
              <Copy size={12} color="var(--muted-text)" />
            </button>
            <a href={dep.hf_repo_url} target="_blank" rel="noopener noreferrer" style={{ width: 26, height: 26, borderRadius: 6, border: '1px solid var(--border-col)', background: 'transparent', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, textDecoration: 'none' }}>
              <ExternalLink size={12} color="var(--muted-text)" />
            </a>
          </div>
        )}

        {/* Code snippet */}
        {isActive && (
          <>
            <button onClick={() => setShowCode(v => !v)} style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '5px 10px', borderRadius: 7, border: '1px solid var(--border-col)', background: 'transparent', cursor: 'pointer', fontSize: 12, color: 'var(--sub)', fontFamily: 'inherit', marginBottom: showCode ? 10 : 0 }}>
              <Code2 size={12} /> {showCode ? 'Hide' : 'Show'} code snippet
            </button>
            {showCode && (
              <div style={{ borderRadius: 10, overflow: 'hidden', border: '1px solid var(--border-col)' }}>
                <div style={{ display: 'flex', background: 'var(--surf2)', borderBottom: '1px solid var(--border-col)' }}>
                  {(['Python', 'cURL', 'JavaScript'] as const).map(t => (
                    <button key={t} onClick={() => setCodeTab(t)} style={{ padding: '7px 14px', border: 'none', background: 'transparent', cursor: 'pointer', fontSize: 11, fontWeight: codeTab === t ? 700 : 400, color: codeTab === t ? 'var(--acc)' : 'var(--sub)', fontFamily: 'inherit', borderBottom: `2px solid ${codeTab === t ? 'var(--acc)' : 'transparent'}' ` }}>
                      {t}
                    </button>
                  ))}
                </div>
                <pre style={{ margin: 0, padding: '14px 16px', background: '#0a0a12', fontSize: 11, color: '#9898b8', fontFamily: 'var(--font-mono)', overflowX: 'auto', lineHeight: 1.65 }}>
                  {CODE_SNIPPETS[codeTab](dep.hf_model_id)}
                </pre>
              </div>
            )}
          </>
        )}

        {/* Action buttons */}
        <div style={{ display: 'flex', gap: 8, marginTop: 14 }}>
          {isActive && (
            <button onClick={onTestInChat} style={{ flex: 1, padding: '8px 14px', borderRadius: 9, background: 'var(--acc)', border: 'none', cursor: 'pointer', fontSize: 13, color: '#fff', fontFamily: 'inherit', fontWeight: 600, boxShadow: '0 2px 8px rgba(59,130,246,.25)' }}>
              Test in Chat
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export function DeploymentsScreen({ setScreen }: DeploymentsScreenProps) {
  const toast = useToast();
  const [deployments, setDeployments] = useState<Deployment[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const hasDataRef = useRef(false);
  const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000';

  const fetchDeployments = useCallback(async () => {
    try {
      const res = await fetch(`${baseUrl}/deployments`, { cache: 'no-store' });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const list: Deployment[] = Array.isArray(data) ? data : (data.deployments ?? []);
      setDeployments(list);
      hasDataRef.current = true;
      setError(null);
    } catch (e: any) {
      if (!hasDataRef.current) setError(e?.message ?? 'Failed to load deployments');
    } finally {
      setLoading(false);
    }
  }, [baseUrl]);

  useEffect(() => {
    void fetchDeployments();
    const interval = setInterval(() => void fetchDeployments(), 5000);
    return () => clearInterval(interval);
  }, [fetchDeployments]);

  const activeCount = deployments.filter(d => d.status === 'completed').length;

  return (
    <div style={{ padding: '24px 26px', overflowY: 'auto', height: '100%' }} className="animate-fadeUp">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 22 }}>
        <div>
          <h1 style={{ fontSize: 20, fontWeight: 800, color: 'var(--text)' }}>Deployments</h1>
          <p style={{ fontSize: 13, color: 'var(--sub)', marginTop: 3 }}>{deployments.length} total · {activeCount} live</p>
        </div>
        <button onClick={() => void fetchDeployments()} style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '7px 14px', borderRadius: 8, border: '1px solid var(--border-col)', background: 'var(--surf2)', cursor: 'pointer', fontSize: 13, color: 'var(--sub)', fontFamily: 'inherit' }}>
          <RefreshCw size={13} /> Refresh
        </button>
      </div>

      {loading && (
        <div style={{ padding: '64px 0', textAlign: 'center' }}>
          <Loader2 size={28} color="var(--acc)" style={{ margin: '0 auto 12px', animation: 'spin 1s linear infinite' }} />
          <div style={{ fontSize: 13, color: 'var(--muted-text)' }}>Loading deployments…</div>
        </div>
      )}

      {!loading && error && (
        <div style={{ padding: '20px', borderRadius: 12, background: 'rgba(239,68,68,.08)', border: '1px solid rgba(239,68,68,.2)', color: 'var(--err)', fontSize: 13, marginBottom: 20 }}>
          {error}
        </div>
      )}

      {!loading && deployments.length === 0 && !error && (
        <div style={{ padding: '80px 0', textAlign: 'center' }}>
          <div style={{ width: 56, height: 56, borderRadius: 16, background: 'rgba(59,130,246,.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 16px' }}>
            <Rocket size={26} color="var(--acc)" />
          </div>
          <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--text)', marginBottom: 8 }}>No deployments yet</div>
          <div style={{ fontSize: 13, color: 'var(--sub)', marginBottom: 16 }}>Deploy a fine-tuned checkpoint to HuggingFace Hub to see it here.</div>
          <button onClick={() => setScreen('training')} style={{ padding: '9px 20px', borderRadius: 9, background: 'var(--acc)', border: 'none', cursor: 'pointer', fontSize: 13, color: '#fff', fontFamily: 'inherit', fontWeight: 600 }}>
            Go to Training →
          </button>
        </div>
      )}

      {!loading && deployments.length > 0 && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(440px,1fr))', gap: 16 }}>
          {deployments.map(dep => (
            <DeploymentCard key={dep.id} dep={dep} onTestInChat={() => setScreen('chat')} />
          ))}
        </div>
      )}
    </div>
  );
}
