'use client';

import { useState, useMemo } from 'react';
import { Run, Dataset, SupportedModel, createRun } from '@/lib/api';
import { SimpleRunList } from '@/components/runs/simple-run-list';
import { RunFilters } from '@/components/runs/run-filters';
import { RunDetailView } from '@/components/runs/run-detail-view';
import { InlineTrainingWizard } from '@/components/wizard/inline-training-wizard';
import { Button } from '@/components/ui/button';
import { Rocket } from 'lucide-react';


interface RunsTabProps {
  runs: Run[];
  loading: boolean;
  selectedRunId: number | null;
  onSelectRun: (id: number | null) => void;
  datasets: Dataset[];
  supportedModels: SupportedModel[];
  selectedProjectId: number | null;
  onRunCreated: (run: Run) => void;
  onRefreshRuns: () => void;
  onError: (message: string) => void;
  onSuccess: (message: string) => void;
  onStatus?: (message: string) => void;
}

export function RunsTab({
  runs,
  loading,
  selectedRunId,
  onSelectRun,
  datasets,
  supportedModels,
  selectedProjectId,
  onRunCreated,
  onRefreshRuns,
  onError,
  onSuccess,
  onStatus,
}: RunsTabProps) {
  const [runSubmitting, setRunSubmitting] = useState(false);
  const [isInlineWizardOpen, setIsInlineWizardOpen] = useState(false);

  // Filter and view state
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [recipeFilter, setRecipeFilter] = useState('all');
  const [viewMode, setViewMode] = useState<'cards' | 'table'>('cards');

  const handleRunSubmit = async (payload: any) => {
    setRunSubmitting(true);
    onError('');
    onSuccess('');
    try {
      const run = await createRun(payload);
      onSuccess(`🎉 Run #${run.id} started successfully! Training in progress...`);
      onRunCreated(run);
      setIsInlineWizardOpen(false);
      // Scroll to top to see the new run
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (error) {
      console.error(error);
      onError((error as Error).message);
    } finally {
      setRunSubmitting(false);
    }
  };

  // Filtered runs
  const filteredRuns = useMemo(() => {
    return runs.filter((run) => {
      // Search filter
      const matchesSearch =
        !searchTerm ||
        run.id.toString().includes(searchTerm) ||
        run.recipe_type.toLowerCase().includes(searchTerm.toLowerCase());

      // Status filter
      const matchesStatus = statusFilter === 'all' || run.status === statusFilter;

      // Recipe filter
      const matchesRecipe = recipeFilter === 'all' || run.recipe_type === recipeFilter;

      return matchesSearch && matchesStatus && matchesRecipe;
    });
  }, [runs, searchTerm, statusFilter, recipeFilter]);

  // Show loading state during run creation
  if (runSubmitting) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="text-center space-y-4">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto"></div>
          <p className="text-muted-foreground">Loading run details...</p>
        </div>
      </div>
    );
   }

  // Show detail view when a run is selected
  if (selectedRunId) {
    return (
      <RunDetailView
        runId={selectedRunId}
        onBack={() => onSelectRun(null)}
        onError={onError}
        onSuccess={onSuccess}
      />
    );
  }

  return (
    <div className="space-y-6">
      {/* Header with New Run button */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Training Runs</h2>
          <p className="text-sm text-muted-foreground">Monitor and manage your training jobs</p>
        </div>
        <Button onClick={() => setIsInlineWizardOpen(true)} size="lg" className="gap-2">
          <Rocket className="h-5 w-5" />
          Start New Training
        </Button>
      </div>

      {/* Inline Wizard */}
      <InlineTrainingWizard
        isOpen={isInlineWizardOpen}
        onClose={() => setIsInlineWizardOpen(false)}
        datasets={datasets}
        supportedModels={supportedModels}
        selectedProjectId={selectedProjectId}
        onSubmit={handleRunSubmit}
      />

      {/* Filters */}
      <RunFilters
        searchTerm={searchTerm}
        onSearchChange={setSearchTerm}
        statusFilter={statusFilter}
        onStatusFilterChange={setStatusFilter}
        recipeFilter={recipeFilter}
        onRecipeFilterChange={setRecipeFilter}
        viewMode={viewMode}
        onViewModeChange={setViewMode}
        totalRuns={runs.length}
        filteredCount={filteredRuns.length}
      />

       {/* Runs Display */}
       <SimpleRunList
         runs={filteredRuns}
         loading={loading}
         selectedRunId={selectedRunId}
         onSelectRun={onSelectRun}
       />
    </div>
  );
}