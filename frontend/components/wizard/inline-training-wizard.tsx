'use client';

import { useState, useEffect, useMemo, useRef } from 'react';
import {
  Dataset,
  RecipeType,
  RunCreatePayload,
  SupportedModel,
  getAutoLearningRate,
  getModelRenderers,
} from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import {
  Sparkles, Brain, MessageSquare, Zap, Target, Shuffle,
  FlaskConical, Calculator, Wrench, Users, Loader2,
  ChevronRight, ChevronLeft, X, Wand2, Settings2,
  ChevronDown, ChevronUp, Rocket, Search, Star, Check,
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface InlineTrainingWizardProps {
  isOpen: boolean;
  onClose: () => void;
  datasets: Dataset[];
  supportedModels: SupportedModel[];
  selectedProjectId: number | null;
  onSubmit: (payload: RunCreatePayload) => Promise<void>;
}

const ALL_RECIPES = [
  { value: 'SFT', label: 'Supervised Fine-Tuning', shortLabel: 'SFT', description: 'Train on instruction-response pairs', icon: Zap, popular: true, supported: true, defaultConfig: { learning_rate: 5e-5, batch_size: 4, rank: 64, epochs: 3 } },
  { value: 'CHAT_SL', label: 'Chat Training', shortLabel: 'Chat', description: 'Conversational AI training', icon: MessageSquare, popular: true, supported: true, defaultConfig: { learning_rate: 5e-4, batch_size: 64, rank: 64 } },
  { value: 'MATH_RL', label: 'Math Reasoning RL', shortLabel: 'Math RL', description: 'Mathematical reasoning with RL', icon: Calculator, popular: true, supported: true, defaultConfig: { learning_rate: 1e-5, rank: 32 } },
  { value: 'DPO', label: 'Direct Preference Optimization', shortLabel: 'DPO', description: 'Learn from preference pairs', icon: Target, popular: false, supported: true },
  { value: 'RL', label: 'Reinforcement Learning', shortLabel: 'RL', description: 'General RL training', icon: Brain, popular: false, supported: true },
  { value: 'DISTILLATION', label: 'Model Distillation', shortLabel: 'Distill', description: 'Compress model knowledge', icon: FlaskConical, popular: false, supported: true },
  { value: 'PPO', label: 'Proximal Policy Optimization', shortLabel: 'PPO', description: 'Advanced RL with PPO', icon: Target, popular: false, supported: false, comingSoon: true },
  { value: 'GRPO', label: 'Group Relative Policy Optimization', shortLabel: 'GRPO', description: 'Group-based RL optimization', icon: Users, popular: false, supported: false, comingSoon: true },
  { value: 'PROMPT_DISTILLATION', label: 'Prompt Distillation', shortLabel: 'P-Distill', description: 'Distill prompt strategies', icon: Sparkles, popular: false, supported: false, comingSoon: true },
  { value: 'TOOL_USE', label: 'Tool Use Training', shortLabel: 'Tool Use', description: 'Train to use external tools', icon: Wrench, popular: false, supported: false, comingSoon: true },
  { value: 'MULTIPLAYER_RL', label: 'Multi-Agent RL', shortLabel: 'Multi-RL', description: 'Multiple agents training', icon: Users, popular: false, supported: false, comingSoon: true },
];

const RECOMMENDED_MODELS = [
  'meta-llama/Llama-3.1-8B-Instruct',
  'Qwen/Qwen3-8B-Base',
  'meta-llama/Llama-3.2-3B',
];

// Extract parameter size from model name (e.g. "8B", "70B", "0.5B")
function extractParamSize(name: string): string | null {
  const m = name.match(/(\d+(?:\.\d+)?)\s*[Bb](?:\b|-)/);
  return m ? `${m[1]}B` : null;
}

// Group models by organisation prefix (before the first "/")
function groupByOrg(models: SupportedModel[]): Record<string, SupportedModel[]> {
  const groups: Record<string, SupportedModel[]> = {};
  for (const m of models) {
    const org = m.model_name.includes('/') ? m.model_name.split('/')[0] : 'Other';
    if (!groups[org]) groups[org] = [];
    groups[org].push(m);
  }
  return groups;
}

// Pretty org names
const ORG_LABELS: Record<string, string> = {
  'meta-llama': 'Meta Llama',
  'Qwen': 'Qwen',
  'mistralai': 'Mistral AI',
  'deepseek-ai': 'DeepSeek',
  'google': 'Google',
  'microsoft': 'Microsoft',
  'Other': 'Other',
};
function orgLabel(org: string) {
  return ORG_LABELS[org] ?? org;
}

export function InlineTrainingWizard({
  isOpen,
  onClose,
  datasets,
  supportedModels,
  selectedProjectId,
  onSubmit,
}: InlineTrainingWizardProps) {
  const [step, setStep] = useState(1);
  const [selectedRecipe, setSelectedRecipe] = useState<RecipeType | ''>('');
  const [selectedModel, setSelectedModel] = useState('meta-llama/Llama-3.1-8B-Instruct');
  const [selectedDataset, setSelectedDataset] = useState('none');
  const [showAllRecipes, setShowAllRecipes] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [modelSearch, setModelSearch] = useState('');

  const [customHyperparams, setCustomHyperparams] = useState('');
  const [selectedRenderer, setSelectedRenderer] = useState('');
  const [availableRenderers, setAvailableRenderers] = useState<string[]>([]);
  const [wandbProject, setWandbProject] = useState('');
  const [lrScheduler, setLrScheduler] = useState('constant');
  const [warmupSteps, setWarmupSteps] = useState(0);

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isCalculatingLr, setIsCalculatingLr] = useState(false);

  const searchRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isOpen) {
      setStep(1);
      setSelectedRecipe('');
      setSelectedModel('meta-llama/Llama-3.1-8B-Instruct');
      setSelectedDataset('none');
      setShowAllRecipes(false);
      setShowAdvanced(false);
      setModelSearch('');
      setCustomHyperparams('');
      setWandbProject('');
      setLrScheduler('constant');
      setWarmupSteps(0);
    }
  }, [isOpen]);

  // Focus search when reaching step 2
  useEffect(() => {
    if (step === 2) setTimeout(() => searchRef.current?.focus(), 100);
  }, [step]);

  useEffect(() => {
    if (selectedModel && step >= 2) {
      getModelRenderers(selectedModel)
        .then((response) => {
          setAvailableRenderers(response.recommended_renderers);
          setSelectedRenderer(response.default_renderer || '');
        })
        .catch(() => {
          setAvailableRenderers([]);
          setSelectedRenderer('');
        });
    }
  }, [selectedModel, step]);

  const handleAutoLr = async () => {
    if (!selectedModel) return;
    setIsCalculatingLr(true);
    try {
      const lr = await getAutoLearningRate(selectedModel, true);
      const current = customHyperparams ? JSON.parse(customHyperparams) : {};
      setCustomHyperparams(JSON.stringify({ ...current, learning_rate: lr }, null, 2));
    } catch (e) {
      console.error('Auto LR failed:', e);
    } finally {
      setIsCalculatingLr(false);
    }
  };

  const handleSubmit = async () => {
    if (!selectedProjectId || !selectedRecipe || !selectedModel) return;
    setIsSubmitting(true);
    try {
      const recipe = ALL_RECIPES.find((r) => r.value === selectedRecipe);
      let hyperparameters: Record<string, unknown> = { ...recipe?.defaultConfig };
      if (customHyperparams.trim()) {
        try {
          hyperparameters = { ...hyperparameters, ...JSON.parse(customHyperparams) };
        } catch {
          alert('Invalid JSON in custom hyperparameters');
          setIsSubmitting(false);
          return;
        }
      }
      if (selectedRenderer) hyperparameters.renderer_name = selectedRenderer;
      if (wandbProject.trim()) hyperparameters.wandb_project = wandbProject.trim();
      if (lrScheduler !== 'constant') hyperparameters.lr_schedule = lrScheduler;
      if (warmupSteps > 0) hyperparameters.lr_warmup_steps = warmupSteps;

      const payload: RunCreatePayload = {
        project_id: selectedProjectId,
        recipe_type: selectedRecipe,
        config_json: { base_model: selectedModel, hyperparameters },
      };
      if (selectedDataset && selectedDataset !== 'none') {
        payload.dataset_id = parseInt(selectedDataset);
      }
      await onSubmit(payload);
      onClose();
    } catch (e) {
      console.error('Error creating run:', e);
    } finally {
      setIsSubmitting(false);
    }
  };

  // ── Model filtering & grouping ─────────────────────────────────────────────

  const filteredModels = useMemo(() => {
    const q = modelSearch.trim().toLowerCase();
    if (!q) return supportedModels;
    return supportedModels.filter(
      (m) =>
        m.model_name.toLowerCase().includes(q) ||
        (m.description ?? '').toLowerCase().includes(q)
    );
  }, [supportedModels, modelSearch]);

  const recommendedFiltered = useMemo(
    () => filteredModels.filter((m) => RECOMMENDED_MODELS.includes(m.model_name)),
    [filteredModels]
  );

  const groupedModels = useMemo(() => groupByOrg(filteredModels), [filteredModels]);

  const popularRecipes = ALL_RECIPES.filter((r) => r.popular);
  const displayedRecipes = showAllRecipes ? ALL_RECIPES : popularRecipes;
  const selectedRecipeInfo = ALL_RECIPES.find((r) => r.value === selectedRecipe);

  if (!isOpen) return null;

  return (
    <div className="mb-6 rounded-xl border border-[var(--border-col)] bg-[var(--surf)] shadow-2xl animate-in slide-in-from-top duration-300 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-[var(--border-col)] px-5 py-4"
        style={{ background: 'linear-gradient(135deg, var(--surf2) 0%, var(--surf) 100%)' }}>
        <div className="flex items-center gap-3">
          <div className="rounded-lg p-2" style={{ background: 'var(--acc)', opacity: 0.9 }}>
            <Rocket className="h-5 w-5 text-white" />
          </div>
          <div>
            <h2 className="text-base font-bold" style={{ color: 'var(--text)' }}>New Training Run</h2>
            <p className="text-xs" style={{ color: 'var(--sub)' }}>
              Step {step} of 2 — {step === 1 ? 'Choose training type' : 'Select model & dataset'}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {/* Step pills */}
          <div className="flex items-center gap-1.5">
            {[1, 2].map((s) => (
              <div
                key={s}
                className="h-2 rounded-full transition-all duration-300"
                style={{
                  width: s === step ? '24px' : '8px',
                  background: s <= step ? 'var(--acc)' : 'var(--border-col)',
                }}
              />
            ))}
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 transition-colors hover:bg-[var(--surf3)]"
          >
            <X className="h-4 w-4" style={{ color: 'var(--muted-text)' }} />
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="p-5">
        {/* ── Step 1: Recipe ─────────────────────────────────────────────── */}
        {step === 1 && (
          <div className="space-y-4">
            <h3 className="text-sm font-semibold" style={{ color: 'var(--sub)' }}>
              SELECT TRAINING TYPE
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5">
              {displayedRecipes.map((recipe) => {
                const RecipeIcon = recipe.icon;
                const isSelected = selectedRecipe === recipe.value;
                const isDisabled = recipe.comingSoon || !recipe.supported;
                return (
                  <button
                    key={recipe.value}
                    onClick={() => !isDisabled && setSelectedRecipe(recipe.value as RecipeType)}
                    disabled={isDisabled}
                    style={isSelected ? {
                      borderColor: 'var(--acc)',
                      background: 'color-mix(in srgb, var(--acc) 10%, var(--surf2))',
                      boxShadow: '0 0 0 1px var(--acc)',
                    } : undefined}
                    className={cn(
                      'group relative rounded-lg border p-3.5 text-left transition-all duration-200',
                      !isDisabled && 'hover:shadow-md cursor-pointer',
                      !isSelected && 'border-[var(--border-col)] bg-[var(--surf2)] hover:border-[var(--acc)/50]',
                      isDisabled && 'opacity-50 cursor-not-allowed'
                    )}
                  >
                    <div className="flex items-start gap-2.5">
                      <div
                        className="rounded-md p-1.5 transition-colors"
                        style={{
                          background: isSelected ? 'var(--acc)' : 'var(--surf3)',
                          color: isSelected ? '#fff' : 'var(--sub)',
                        }}
                      >
                        <RecipeIcon className="h-4 w-4" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-1.5">
                          <span className="font-semibold text-xs" style={{ color: 'var(--text)' }}>
                            {recipe.shortLabel}
                          </span>
                          {recipe.comingSoon && (
                            <span className="text-[9px] px-1 py-0 rounded border"
                              style={{ borderColor: 'var(--border-col)', color: 'var(--muted-text)' }}>
                              Soon
                            </span>
                          )}
                        </div>
                        <p className="text-[11px] mt-0.5 line-clamp-2" style={{ color: 'var(--muted-text)' }}>
                          {recipe.description}
                        </p>
                      </div>
                    </div>
                    {isSelected && (
                      <div className="absolute top-2 right-2 h-2 w-2 rounded-full animate-pulse"
                        style={{ background: 'var(--acc)' }} />
                    )}
                  </button>
                );
              })}
            </div>

            <button
              onClick={() => setShowAllRecipes((v) => !v)}
              className="w-full rounded-lg border py-2 text-xs font-medium flex items-center justify-center gap-1.5 transition-colors"
              style={{ borderColor: 'var(--border-col)', color: 'var(--sub)' }}
            >
              {showAllRecipes ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
              {showAllRecipes ? 'Show Popular Only' : `Show All ${ALL_RECIPES.length} Recipe Types`}
            </button>
          </div>
        )}

        {/* ── Step 2: Model & Dataset ─────────────────────────────────────── */}
        {step === 2 && (
          <div className="space-y-5">
            {/* Model Section */}
            <div>
              <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--sub)' }}>
                SELECT BASE MODEL
              </h3>

              {/* Search */}
              <div className="relative mb-3">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5"
                  style={{ color: 'var(--muted-text)' }} />
                <input
                  ref={searchRef}
                  value={modelSearch}
                  onChange={(e) => setModelSearch(e.target.value)}
                  placeholder="Search models..."
                  className="w-full rounded-lg border pl-9 pr-3 py-2 text-sm bg-transparent outline-none focus:ring-1"
                  style={{
                    borderColor: 'var(--border-col)',
                    color: 'var(--text)',
                    caretColor: 'var(--acc)',
                  }}
                  onFocus={(e) => (e.target.style.borderColor = 'var(--acc)')}
                  onBlur={(e) => (e.target.style.borderColor = 'var(--border-col)')}
                />
                {modelSearch && (
                  <button
                    onClick={() => setModelSearch('')}
                    className="absolute right-3 top-1/2 -translate-y-1/2"
                    style={{ color: 'var(--muted-text)' }}
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                )}
              </div>

              {/* Model list — scrollable */}
              <div className="rounded-lg border overflow-hidden" style={{ borderColor: 'var(--border-col)' }}>
                <div className="max-h-72 overflow-y-auto">
                  {filteredModels.length === 0 && (
                    <div className="py-8 text-center text-sm" style={{ color: 'var(--muted-text)' }}>
                      No models match "{modelSearch}"
                    </div>
                  )}

                  {/* Recommended section */}
                  {recommendedFiltered.length > 0 && (
                    <ModelGroup
                      label="Recommended"
                      icon={<Star className="h-3 w-3" style={{ color: 'var(--warn)' }} />}
                      models={recommendedFiltered}
                      selectedModel={selectedModel}
                      onSelect={setSelectedModel}
                      accentColor="var(--warn)"
                      recommended
                    />
                  )}

                  {/* Other groups */}
                  {Object.entries(groupedModels)
                    .sort(([a], [b]) => orgLabel(a).localeCompare(orgLabel(b)))
                    .map(([org, models]) => (
                      <ModelGroup
                        key={org}
                        label={orgLabel(org)}
                        models={models}
                        selectedModel={selectedModel}
                        onSelect={setSelectedModel}
                      />
                    ))}
                </div>
              </div>

              {/* Selected model summary */}
              {selectedModel && (
                <div className="mt-2.5 flex items-center gap-2 px-3 py-2 rounded-lg text-xs"
                  style={{ background: 'color-mix(in srgb, var(--acc) 8%, var(--surf2))', color: 'var(--sub)' }}>
                  <Check className="h-3.5 w-3.5 flex-shrink-0" style={{ color: 'var(--acc)' }} />
                  <span className="truncate font-medium" style={{ color: 'var(--text)' }}>{selectedModel}</span>
                  {extractParamSize(selectedModel) && (
                    <span className="ml-auto flex-shrink-0 rounded px-1.5 py-0.5 font-mono font-bold text-[10px]"
                      style={{ background: 'var(--acc)', color: '#fff' }}>
                      {extractParamSize(selectedModel)}
                    </span>
                  )}
                </div>
              )}
            </div>

            {/* Dataset */}
            <div>
              <h3 className="text-sm font-semibold mb-2" style={{ color: 'var(--sub)' }}>
                DATASET <span className="font-normal" style={{ color: 'var(--muted-text)' }}>(optional)</span>
              </h3>
              <Select value={selectedDataset} onValueChange={setSelectedDataset}>
                <SelectTrigger>
                  <SelectValue placeholder="Choose dataset..." />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">No dataset (use defaults)</SelectItem>
                  {datasets.map((ds) => (
                    <SelectItem key={ds.id} value={ds.id.toString()}>
                      {ds.name} ({ds.kind})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Smart Defaults */}
            {selectedRecipeInfo?.defaultConfig && (
              <div className="rounded-lg px-4 py-3 border"
                style={{ borderColor: 'var(--border-col)', background: 'var(--surf2)' }}>
                <div className="flex items-center gap-2 mb-2">
                  <Sparkles className="h-3.5 w-3.5" style={{ color: 'var(--warn)' }} />
                  <span className="text-xs font-semibold" style={{ color: 'var(--text)' }}>
                    Smart Defaults — {selectedRecipeInfo.shortLabel}
                  </span>
                </div>
                <div className="flex flex-wrap gap-x-4 gap-y-1">
                  {Object.entries(selectedRecipeInfo.defaultConfig).map(([k, v]) => (
                    <span key={k} className="text-xs" style={{ color: 'var(--sub)' }}>
                      <span style={{ color: 'var(--muted-text)' }}>{k}:</span>{' '}
                      <span className="font-mono font-medium" style={{ color: 'var(--text)' }}>{String(v)}</span>
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Advanced toggle */}
            <button
              onClick={() => setShowAdvanced((v) => !v)}
              className="w-full flex items-center justify-center gap-1.5 rounded-lg border py-2 text-xs font-medium transition-colors"
              style={{ borderColor: 'var(--border-col)', color: 'var(--sub)' }}
            >
              <Settings2 className="h-3.5 w-3.5" />
              {showAdvanced ? 'Hide' : 'Show'} Advanced Settings
              {showAdvanced ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
            </button>

            {showAdvanced && (
              <div className="space-y-4 rounded-lg border p-4"
                style={{ borderColor: 'var(--border-col)', background: 'var(--surf2)' }}>
                {availableRenderers.length > 0 && (
                  <div>
                    <Label>Renderer</Label>
                    <Select value={selectedRenderer} onValueChange={setSelectedRenderer}>
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {availableRenderers.map((r) => (
                          <SelectItem key={r} value={r}>{r}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                )}
                <div>
                  <Label>Weights & Biases Project (Optional)</Label>
                  <Input placeholder="my-project" value={wandbProject} onChange={(e) => setWandbProject(e.target.value)} />
                </div>
                <div>
                  <Label>Learning Rate Schedule</Label>
                  <Select value={lrScheduler} onValueChange={setLrScheduler}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="constant">Constant</SelectItem>
                      <SelectItem value="linear">Linear Decay</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label>Warmup Steps</Label>
                  <Input type="number" min="0" placeholder="0" value={warmupSteps}
                    onChange={(e) => setWarmupSteps(parseInt(e.target.value) || 0)} />
                </div>
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <Label>Custom Hyperparameters (JSON)</Label>
                    <button
                      onClick={handleAutoLr}
                      disabled={isCalculatingLr}
                      className="flex items-center gap-1 text-xs rounded px-2 py-1 transition-colors"
                      style={{ color: 'var(--acc)', background: 'color-mix(in srgb, var(--acc) 10%, transparent)' }}
                    >
                      {isCalculatingLr
                        ? <Loader2 className="h-3 w-3 animate-spin" />
                        : <Wand2 className="h-3 w-3" />}
                      Auto LR
                    </button>
                  </div>
                  <textarea
                    className="w-full rounded-md border px-3 py-2 text-xs font-mono bg-transparent"
                    style={{ borderColor: 'var(--border-col)', color: 'var(--text)', minHeight: '100px' }}
                    placeholder={'{\n  "learning_rate": 5e-5,\n  "epochs": 3\n}'}
                    value={customHyperparams}
                    onChange={(e) => setCustomHyperparams(e.target.value)}
                  />
                </div>
              </div>
            )}
          </div>
        )}

        {/* Navigation */}
        <div className="flex items-center justify-between mt-5 pt-4 border-t"
          style={{ borderColor: 'var(--border-col)' }}>
          <div className="flex gap-2">
            {step > 1 && (
              <button
                onClick={() => setStep(step - 1)}
                className="flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-sm transition-colors hover:bg-[var(--surf3)]"
                style={{ borderColor: 'var(--border-col)', color: 'var(--sub)' }}
              >
                <ChevronLeft className="h-4 w-4" /> Back
              </button>
            )}
            <button
              onClick={onClose}
              className="rounded-lg px-3 py-1.5 text-sm transition-colors hover:bg-[var(--surf3)]"
              style={{ color: 'var(--muted-text)' }}
            >
              Cancel
            </button>
          </div>

          {step === 1 ? (
            <button
              onClick={() => setStep(2)}
              disabled={!selectedRecipe}
              className="flex items-center gap-1.5 rounded-lg px-4 py-1.5 text-sm font-semibold transition-all disabled:opacity-40"
              style={{ background: 'var(--acc)', color: '#fff' }}
            >
              Continue <ChevronRight className="h-4 w-4" />
            </button>
          ) : (
            <button
              onClick={handleSubmit}
              disabled={isSubmitting || !selectedRecipe || !selectedModel}
              className="flex items-center gap-1.5 rounded-lg px-4 py-1.5 text-sm font-semibold transition-all disabled:opacity-40"
              style={{ background: 'var(--acc)', color: '#fff' }}
            >
              {isSubmitting ? (
                <><Loader2 className="h-4 w-4 animate-spin" /> Starting…</>
              ) : (
                <><Rocket className="h-4 w-4" /> Start Training</>
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

// ── ModelGroup sub-component ────────────────────────────────────────────────

interface ModelGroupProps {
  label: string;
  icon?: React.ReactNode;
  models: SupportedModel[];
  selectedModel: string;
  onSelect: (name: string) => void;
  accentColor?: string;
  recommended?: boolean;
}

function ModelGroup({ label, icon, models, selectedModel, onSelect, accentColor, recommended }: ModelGroupProps) {
  return (
    <div>
      {/* Sticky category header */}
      <div
        className="sticky top-0 z-10 flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wider"
        style={{
          background: recommended
            ? 'color-mix(in srgb, var(--warn) 8%, var(--surf))'
            : 'var(--surf2)',
          color: accentColor ?? 'var(--muted-text)',
          borderBottom: '1px solid var(--border-col)',
        }}
      >
        {icon}
        {label}
        <span className="ml-auto font-normal normal-case tracking-normal" style={{ color: 'var(--muted-text)' }}>
          {models.length}
        </span>
      </div>

      {models.map((model) => {
        const isSelected = selectedModel === model.model_name;
        const paramSize = extractParamSize(model.model_name);
        const shortName = model.model_name.includes('/')
          ? model.model_name.split('/')[1]
          : model.model_name;

        return (
          <ModelRow
            key={model.model_name}
            model={model}
            shortName={shortName}
            paramSize={paramSize}
            isSelected={isSelected}
            onSelect={onSelect}
          />
        );
      })}
    </div>
  );
}

// ── ModelRow sub-component ──────────────────────────────────────────────────

interface ModelRowProps {
  model: SupportedModel;
  shortName: string;
  paramSize: string | null;
  isSelected: boolean;
  onSelect: (name: string) => void;
}

function ModelRow({ model, shortName, paramSize, isSelected, onSelect }: ModelRowProps) {
  const [hovered, setHovered] = useState(false);

  return (
    <button
      onClick={() => onSelect(model.model_name)}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className="w-full flex items-center gap-3 px-3 py-2.5 text-left transition-all duration-150 relative overflow-hidden"
      style={{
        background: isSelected
          ? 'color-mix(in srgb, var(--acc) 12%, var(--surf))'
          : hovered
          ? 'var(--surf2)'
          : 'var(--surf)',
        borderLeft: isSelected ? '3px solid var(--acc)' : '3px solid transparent',
        borderBottom: '1px solid var(--border-col)',
      }}
    >
      {/* Shimmer on hover */}
      {(hovered || isSelected) && (
        <span
          className="pointer-events-none absolute inset-0"
          style={{
            background: isSelected
              ? 'linear-gradient(90deg, transparent, color-mix(in srgb, var(--acc) 6%, transparent), transparent)'
              : 'linear-gradient(90deg, transparent, color-mix(in srgb, var(--text) 3%, transparent), transparent)',
            animation: 'shimmer 1.4s infinite',
          }}
        />
      )}

      {/* Selection indicator */}
      <div
        className="flex-shrink-0 h-4 w-4 rounded-full border-2 flex items-center justify-center transition-all"
        style={{
          borderColor: isSelected ? 'var(--acc)' : 'var(--border-col)',
          background: isSelected ? 'var(--acc)' : 'transparent',
        }}
      >
        {isSelected && <Check className="h-2.5 w-2.5 text-white" strokeWidth={3} />}
      </div>

      {/* Model info */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span
            className="text-sm font-medium truncate"
            style={{ color: isSelected ? 'var(--acc)' : 'var(--text)' }}
          >
            {shortName}
          </span>
          {paramSize && (
            <span
              className="flex-shrink-0 text-[10px] font-mono font-bold px-1.5 py-0.5 rounded"
              style={{
                background: isSelected
                  ? 'var(--acc)'
                  : 'color-mix(in srgb, var(--acc) 15%, var(--surf3))',
                color: isSelected ? '#fff' : 'var(--acc)',
              }}
            >
              {paramSize}
            </span>
          )}
        </div>
        {model.description && (
          <p className="text-[11px] truncate mt-0.5" style={{ color: 'var(--muted-text)' }}>
            {model.description}
          </p>
        )}
      </div>
    </button>
  );
}
