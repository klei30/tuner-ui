'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Copy, Trash2, Settings2, ChevronDown, Download, Sparkles, User, Bot } from 'lucide-react';
import { useToast } from '@/components/ui/toast-provider';
import { chatWithModel } from '@/lib/api';
import type { SupportedModel, RegisteredModel } from '@/lib/api';

type Screen = 'overview' | 'training' | 'models' | 'chat' | 'datasets' | 'deployments';

interface ChatScreenProps {
  supportedModels: SupportedModel[];
  registeredModels: RegisteredModel[];
  setScreen: (s: Screen) => void;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

const SUGGESTIONS = [
  'Explain fine-tuning in simple terms',
  'What is LoRA and when should I use it?',
  'Write a Python function to load a dataset',
  'Compare SFT vs DPO training',
];

export function ChatScreen({ supportedModels, registeredModels, setScreen }: ChatScreenProps) {
  const toast = useToast();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState('');
  const [showParams, setShowParams] = useState(false);
  const [systemPrompt, setSystemPrompt] = useState('You are a helpful AI assistant.');
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(512);
  const [topP, setTopP] = useState(0.9);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const allModels = [
    ...supportedModels.map(m => ({ value: m.model_name, label: m.model_name.split('/')[1] ?? m.model_name, type: 'Base' })),
    ...registeredModels.map(m => ({ value: `tinker://${m.name}`, label: m.name, type: 'Fine-tuned' })),
  ];

  useEffect(() => {
    if (!selectedModel && allModels.length > 0) setSelectedModel(allModels[0].value);
  }, [allModels.length]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const selectedModelLabel = allModels.find(m => m.value === selectedModel)?.label ?? selectedModel;

  async function send(text?: string) {
    const content = text ?? input.trim();
    if (!content || loading) return;
    if (!selectedModel) { toast('Select a model first', 'err'); return; }

    const userMsg: Message = { role: 'user', content, timestamp: new Date() };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const isTinker = selectedModel.startsWith('tinker://');
      const resp = await chatWithModel({
        prompt: content,
        base_model: isTinker ? undefined : selectedModel,
        model_id: isTinker ? undefined : undefined,
        temperature,
        max_tokens: maxTokens,
      });
      setMessages(prev => [...prev, { role: 'assistant', content: resp.response, timestamp: new Date() }]);
    } catch (e: any) {
      toast(e?.message ?? 'Chat failed', 'err');
      setMessages(prev => prev.slice(0, -1));
    } finally {
      setLoading(false);
    }
  }

  function exportJSONL() {
    const lines = messages.map(m => JSON.stringify({ role: m.role, content: m.content })).join('\n');
    const blob = new Blob([lines], { type: 'application/jsonl' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'chat-export.jsonl'; a.click();
    URL.revokeObjectURL(url);
    toast('Exported as JSONL', 'ok');
  }

  return (
    <div style={{ display: 'flex', height: '100%', overflow: 'hidden' }}>
      {/* Chat area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* Top bar */}
        <div style={{ padding: '14px 20px', borderBottom: '1px solid var(--border-col)', display: 'flex', alignItems: 'center', gap: 10, flexShrink: 0 }}>
          <div style={{ flex: 1 }}>
            <h1 style={{ fontSize: 16, fontWeight: 800, color: 'var(--text)' }}>Chat Playground</h1>
          </div>
          {/* Model selector */}
          <div style={{ position: 'relative' }}>
            <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)} style={{ padding: '7px 32px 7px 12px', borderRadius: 9, border: '1.5px solid var(--border-col)', background: 'var(--surf2)', color: 'var(--text)', fontSize: 13, fontFamily: 'inherit', outline: 'none', cursor: 'pointer', appearance: 'none', minWidth: 180 }}>
              {allModels.length === 0 ? <option value="">No models available</option> : allModels.map(m => <option key={m.value} value={m.value}>[{m.type}] {m.label}</option>)}
            </select>
            <ChevronDown size={12} style={{ position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--muted-text)', pointerEvents: 'none' }} />
          </div>
          <button onClick={() => setShowParams(v => !v)} style={{ padding: '7px 12px', borderRadius: 8, border: `1.5px solid ${showParams ? 'var(--acc)' : 'var(--border-col)'}`, background: showParams ? 'rgba(59,130,246,.08)' : 'var(--surf2)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 5, fontSize: 12, color: showParams ? 'var(--acc)' : 'var(--sub)', fontFamily: 'inherit' }}>
            <Settings2 size={13} /> Params
          </button>
          {messages.length > 0 && (
            <>
              <button onClick={exportJSONL} style={{ padding: '7px 12px', borderRadius: 8, border: '1px solid var(--border-col)', background: 'var(--surf2)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 5, fontSize: 12, color: 'var(--sub)', fontFamily: 'inherit' }}>
                <Download size={13} /> Export
              </button>
              <button onClick={() => setMessages([])} style={{ width: 32, height: 32, borderRadius: 8, border: '1px solid var(--border-col)', background: 'var(--surf2)', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Trash2 size={13} color="var(--muted-text)" />
              </button>
            </>
          )}
        </div>

        {/* Messages */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '20px 24px', display: 'flex', flexDirection: 'column', gap: 16 }}>
          {messages.length === 0 ? (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 20 }}>
              <div style={{ width: 56, height: 56, borderRadius: 16, background: 'linear-gradient(135deg,var(--acc),#8b5cf6)', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 8px 24px rgba(59,130,246,.35)' }}>
                <Sparkles size={26} color="#fff" />
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text)', marginBottom: 6 }}>Start a conversation</div>
                <div style={{ fontSize: 13, color: 'var(--sub)' }}>Testing <strong style={{ color: 'var(--text)' }}>{selectedModelLabel}</strong> · Try a suggestion below</div>
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center', maxWidth: 480 }}>
                {SUGGESTIONS.map(s => (
                  <button key={s} onClick={() => send(s)} style={{ padding: '8px 14px', borderRadius: 999, border: '1.5px solid var(--border-col)', background: 'var(--surf2)', cursor: 'pointer', fontSize: 12, color: 'var(--sub)', fontFamily: 'inherit', transition: 'all .15s' }}
                    onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--acc)'; e.currentTarget.style.color = 'var(--acc)'; }}
                    onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border-col)'; e.currentTarget.style.color = 'var(--sub)'; }}>
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((msg, i) => (
              <div key={i} style={{ display: 'flex', gap: 12, alignItems: 'flex-start', flexDirection: msg.role === 'user' ? 'row-reverse' : 'row' }}>
                <div style={{ width: 32, height: 32, borderRadius: 10, background: msg.role === 'user' ? 'var(--acc)' : 'linear-gradient(135deg,var(--acc),#8b5cf6)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                  {msg.role === 'user' ? <User size={15} color="#fff" /> : <Bot size={15} color="#fff" />}
                </div>
                <div style={{ maxWidth: '75%' }}>
                  <div style={{ padding: '12px 16px', borderRadius: msg.role === 'user' ? '14px 4px 14px 14px' : '4px 14px 14px 14px', background: msg.role === 'user' ? 'var(--acc)' : 'var(--surf2)', color: msg.role === 'user' ? '#fff' : 'var(--text)', fontSize: 14, lineHeight: 1.65, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                    {msg.content}
                  </div>
                  <div style={{ fontSize: 10, color: 'var(--muted-text)', marginTop: 4, textAlign: msg.role === 'user' ? 'right' : 'left' }}>
                    {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                </div>
                <button onClick={() => { navigator.clipboard.writeText(msg.content); toast('Copied', 'ok'); }} style={{ opacity: 0, width: 26, height: 26, borderRadius: 6, border: '1px solid var(--border-col)', background: 'var(--surf2)', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'opacity .15s', flexShrink: 0, alignSelf: 'flex-start', marginTop: 4 }}
                  onMouseEnter={e => (e.currentTarget.style.opacity = '1')} onMouseLeave={e => (e.currentTarget.style.opacity = '0')}>
                  <Copy size={11} color="var(--muted-text)" />
                </button>
              </div>
            ))
          )}
          {loading && (
            <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
              <div style={{ width: 32, height: 32, borderRadius: 10, background: 'linear-gradient(135deg,var(--acc),#8b5cf6)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                <Bot size={15} color="#fff" />
              </div>
              <div style={{ padding: '14px 18px', borderRadius: '4px 14px 14px 14px', background: 'var(--surf2)' }}>
                <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
                  {[0, 1, 2].map(i => <div key={i} className="animate-pulse-dot" style={{ width: 6, height: 6, borderRadius: 999, background: 'var(--acc)', animationDelay: `${i * .2}s` }} />)}
                </div>
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div style={{ padding: '16px 20px', borderTop: '1px solid var(--border-col)', flexShrink: 0 }}>
          <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end', background: 'var(--surf2)', border: '1.5px solid var(--border-col)', borderRadius: 14, padding: '10px 14px', transition: 'border-color .15s' }}
            onFocusCapture={e => (e.currentTarget.style.borderColor = 'var(--acc)')}
            onBlurCapture={e => (e.currentTarget.style.borderColor = 'var(--border-col)')}>
            <textarea ref={textareaRef} value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }} placeholder="Message the model… (Enter to send, Shift+Enter for newline)" rows={1} style={{ flex: 1, background: 'transparent', border: 'none', outline: 'none', resize: 'none', fontSize: 14, color: 'var(--text)', fontFamily: 'inherit', lineHeight: 1.5, minHeight: 22, maxHeight: 160, overflowY: 'auto' }} />
            <button onClick={() => send()} disabled={!input.trim() || loading} style={{ width: 34, height: 34, borderRadius: 10, background: input.trim() && !loading ? 'var(--acc)' : 'var(--surf3)', border: 'none', cursor: input.trim() && !loading ? 'pointer' : 'default', display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'background .15s', flexShrink: 0 }}>
              <Send size={15} color={input.trim() && !loading ? '#fff' : 'var(--muted-text)'} />
            </button>
          </div>
        </div>
      </div>

      {/* Params panel */}
      {showParams && (
        <div style={{ width: 260, borderLeft: '1px solid var(--border-col)', padding: '20px 18px', overflowY: 'auto', flexShrink: 0, background: 'var(--surf)' }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text)', marginBottom: 18 }}>Generation Params</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div>
              <label style={{ fontSize: 11, fontWeight: 700, color: 'var(--sub)', display: 'block', marginBottom: 6 }}>System Prompt</label>
              <textarea value={systemPrompt} onChange={e => setSystemPrompt(e.target.value)} rows={4} style={{ width: '100%', padding: '8px 10px', borderRadius: 9, border: '1.5px solid var(--border-col)', background: 'var(--surf2)', color: 'var(--text)', fontSize: 12, fontFamily: 'inherit', outline: 'none', resize: 'vertical', boxSizing: 'border-box' }}
                onFocus={e => (e.target.style.borderColor = 'var(--acc)')} onBlur={e => (e.target.style.borderColor = 'var(--border-col)')} />
            </div>
            {([
              { label: 'Temperature', value: temperature, setter: setTemperature, min: 0, max: 2, step: 0.1 },
              { label: 'Top P', value: topP, setter: setTopP, min: 0, max: 1, step: 0.05 },
            ] as const).map(({ label, value, setter, min, max, step }) => (
              <div key={label}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                  <label style={{ fontSize: 11, fontWeight: 700, color: 'var(--sub)' }}>{label}</label>
                  <span style={{ fontSize: 11, color: 'var(--text)', fontWeight: 600 }}>{value.toFixed(2)}</span>
                </div>
                <input type="range" min={min} max={max} step={step} value={value} onChange={e => (setter as any)(parseFloat(e.target.value))} style={{ width: '100%', accentColor: 'var(--acc)' }} />
              </div>
            ))}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                <label style={{ fontSize: 11, fontWeight: 700, color: 'var(--sub)' }}>Max Tokens</label>
                <span style={{ fontSize: 11, color: 'var(--text)', fontWeight: 600 }}>{maxTokens}</span>
              </div>
              <input type="range" min={64} max={4096} step={64} value={maxTokens} onChange={e => setMaxTokens(parseInt(e.target.value))} style={{ width: '100%', accentColor: 'var(--acc)' }} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
