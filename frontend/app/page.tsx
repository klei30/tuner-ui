'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Dataset,
  ModelCatalogResponse,
  Project,
  RegisteredModel,
  Run,
  SupportedModel,
  createProject,
  getDatasets,
  getModelCatalog,
  getProjects,
  getRuns,
} from '@/lib/api';
import { DashboardShell } from '@/components/layout/dashboard-shell';
import { Sidebar } from '@/components/layout/sidebar';
import { TopBar } from '@/components/layout/top-bar';
import { StatsGrid } from '@/components/overview/stats-grid';
import { OverviewTab } from '@/components/tabs/overview-tab';
import { RunsTab } from '@/components/tabs/runs-tab';
import { DatasetsTab } from '@/components/tabs/datasets-tab';
import { ModelsTab } from '@/components/tabs/models-tab';
import { ChatTab } from '@/components/tabs/chat-tab';

import { Button } from '@/components/ui/button';

const POLL_INTERVAL_MS = 3000; // Poll every 3 seconds for faster updates

export default function DashboardPage() {
  const [loading, setLoading] = useState(true);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<number | null>(null);

  const [runs, setRuns] = useState<Run[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [modelCatalog, setModelCatalog] = useState<ModelCatalogResponse | null>(null);
  const [activeTab, setActiveTab] = useState('overview');

  const supportedModels = useMemo(() => modelCatalog?.supported_models ?? [], [modelCatalog]);
  const registeredModels = useMemo(() => modelCatalog?.registered_models ?? [], [modelCatalog]);

  const stats = useMemo(
    () => [
      { label: 'Projects', value: projects.length },
      { label: 'Datasets', value: datasets.length },
      { label: 'Total Runs', value: runs.length },
      { label: 'Active Runs', value: runs.filter((run) => run.status === 'running').length },
    ],
    [projects.length, datasets.length, runs]
  );

  const refreshModelCatalog = useCallback(async () => {
    console.log('Fetching model catalog...');
    try {
      const catalog = await getModelCatalog();
      console.log('Model catalog response received');
      console.log('Supported models count:', catalog.supported_models?.length || 0);
      console.log('Registered models count:', catalog.registered_models?.length || 0);
      if (catalog.registered_models) {
        catalog.registered_models.forEach(model => {
          console.log('  Registered model:', model.name, 'ID:', model.id);
        });
      }
      setModelCatalog(catalog);
      console.log('Model catalog state updated');
      console.log('Setting modelCatalog with:', catalog.registered_models?.length, 'registered models');
    } catch (error) {
      console.error('Error fetching model catalog:', error);
      setModelCatalog(null);
      throw error;
    }
  }, []);

  const refreshDatasets = useCallback(async () => {
    const ds = await getDatasets();
    setDatasets(ds);
  }, []);

  const refreshRuns = useCallback(
    async (projectId: number) => {
      const data = await getRuns(projectId);
      setRuns(data);
      if (!data.length) {
        setSelectedRunId(null);
        return;
      }
      // Keep current selection if valid, otherwise don't auto-select
      setSelectedRunId((prev) => (prev && data.some((run) => run.id === prev) ? prev : null));
    },
    []
  );

  useEffect(() => {
    // SECURITY: Only log non-sensitive environment info
    console.log('Environment check:');
    console.log('NEXT_PUBLIC_API_BASE_URL:', process.env.NEXT_PUBLIC_API_BASE_URL);
    console.log('API key configured:', !!process.env.NEXT_PUBLIC_TINKER_API_KEY);

    const bootstrap = async () => {
      console.log('Bootstrap starting...');
      try {
        setLoading(true);
        setErrorMessage(null);

        console.log('Fetching projects...');
        let projectList: Project[];
        try {
          projectList = await getProjects();
          console.log('Projects successful:', projectList.length);
        } catch (err) {
          console.error('Failed to fetch projects!', err);
          throw err;
        }

        if (!projectList.length) {
          console.log('No projects found, creating demo project...');
          const project = await createProject({
            name: 'Demo Project',
            description: 'Auto-generated project for first-time setup.',
          });
          projectList = [project];
          setStatusMessage('Created a demo project automatically.');
        }

        setProjects(projectList);
        const firstProject = projectList[0]?.id ?? null;
        setSelectedProjectId(firstProject);

        console.log('Starting parallel fetches for runs, models, datasets...');
        const results = await Promise.allSettled([
          firstProject ? refreshRuns(firstProject) : Promise.resolve(),
          refreshModelCatalog(),
          refreshDatasets(),
        ]);

        results.forEach((res, i) => {
          if (res.status === 'rejected') {
            console.error(`Fetch ${i} failed:`, res.reason);
          } else {
            console.log(`Fetch ${i} succeeded`);
          }
        });

        console.log('Bootstrap flow concluded');
      } catch (error) {
        console.error('CRITICAL BOOTSTRAP FAILURE:', error);
        setErrorMessage(`Bootstrap failed: ${(error as Error).message}. Check console for details.`);
      } finally {
        setLoading(false);
      }
    };

    void bootstrap();
  }, [refreshDatasets, refreshModelCatalog, refreshRuns]);

  useEffect(() => {
    if (!selectedProjectId) return;
    const interval = setInterval(() => {
      void refreshRuns(selectedProjectId);
    }, POLL_INTERVAL_MS);

    return () => clearInterval(interval);
  }, [refreshRuns, selectedProjectId]);

  const modelOptions = useMemo(() => {
    console.log('Computing modelOptions...');
    console.log('supportedModels count:', supportedModels.length);
    console.log('registeredModels count:', registeredModels.length);

    const options: Array<{ value: string; label: string }> = [];
    for (const model of supportedModels) {
      options.push({ value: `supported::${model.model_name}`, label: `Supported · ${model.model_name}` });
    }
    for (const model of registeredModels) {
      console.log('Adding registered model:', model.name, 'ID:', model.id);
      options.push({ value: `registered::${model.id}`, label: `Registered · ${model.name}` });
    }

    console.log('Total modelOptions:', options.length);
    return options;
  }, [registeredModels, supportedModels]);

  return (
    <DashboardShell
      sidebar={<Sidebar activeTab={activeTab} onTabChange={setActiveTab} />}
      topbar={
        <TopBar
          subtitle="Launch cookbook recipes, monitor training, and chat with checkpoints."
          projects={projects}
          selectedProjectId={selectedProjectId}
          onSelectProject={(projectId) => {
            setSelectedProjectId(projectId);
            void refreshRuns(projectId);
            setErrorMessage(null);
            setSuccessMessage(null);
          }}
          onProjectCreated={(project) => {
            setProjects((prev) => [project, ...prev]);
            setSelectedProjectId(project.id);
            void refreshRuns(project.id);
            setSuccessMessage(`✅ Project "${project.name}" created successfully!`);
            setErrorMessage(null);
          }}
          onError={setErrorMessage}
          rightSlot={
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                if (selectedProjectId) {
                  void refreshRuns(selectedProjectId);
                }
                void refreshModelCatalog();
                void refreshDatasets();
              }}
            >
              Refresh
            </Button>
          }
        />
      }
    >
      {errorMessage ? (
        <div className="mb-4 rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          {errorMessage}
        </div>
      ) : null}
      {statusMessage ? (
        <div className="mb-4 rounded-lg border border-green-500/40 bg-green-500/10 px-4 py-3 text-sm text-green-700 dark:text-green-400">
          {statusMessage}
        </div>
      ) : null}

      <div className="mt-6">
        {activeTab === 'overview' && (
          <OverviewTab
            stats={stats}
            loading={loading}
            onNavigateToRuns={() => setActiveTab('runs')}
            onNavigateToDatasets={() => setActiveTab('datasets')}
            onNavigateToModels={() => setActiveTab('models')}
            onNavigateToChat={() => setActiveTab('chat')}
          />
        )}
        {activeTab === 'runs' && (
          <RunsTab
            runs={runs}
            loading={loading}
            selectedRunId={selectedRunId}
            onSelectRun={setSelectedRunId}
            datasets={datasets}
            supportedModels={supportedModels}
            selectedProjectId={selectedProjectId}
            onRunCreated={(run) => {
              setRuns((prev) => [run, ...prev]);
              if (selectedProjectId) void refreshRuns(selectedProjectId);
            }}
            onRefreshRuns={() => {
              if (selectedProjectId) void refreshRuns(selectedProjectId);
            }}
            onError={setErrorMessage}
            onSuccess={setSuccessMessage}
          />
        )}
        {activeTab === 'datasets' && (
          <DatasetsTab
            datasets={datasets}
            onDatasetCreated={(dataset) => {
              setDatasets((prev) => [dataset, ...prev]);
            }}
            onError={setErrorMessage}
            onSuccess={setStatusMessage}
          />
        )}
        {activeTab === 'models' && (
          <ModelsTab
            modelCatalog={modelCatalog}
            datasets={datasets}
            onError={setErrorMessage}
            onSuccess={(message) => {
              setStatusMessage(message);
              // Refresh model catalog after successful registration
              void refreshModelCatalog();
            }}
            onFineTunePrefill={(model: SupportedModel | RegisteredModel, datasetId?: number) => {
              setStatusMessage(`Prefilled run form for ${'model_name' in model ? model.model_name : model.name}.`);
            }}
            onOpenRunsTab={() => setActiveTab('runs')}
          />
        )}
        {activeTab === 'chat' && (
          <ChatTab
            modelOptions={modelOptions}
            runs={runs}
            onError={setErrorMessage}
          />
        )}
      </div>
    </DashboardShell>
  );
}