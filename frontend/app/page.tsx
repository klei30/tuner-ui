'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import type { Dataset, ModelCatalogResponse, Project, Run } from '@/lib/api';
import { createProject, getDatasets, getModelCatalog, getProjects, getRuns } from '@/lib/api';

import { ToastProvider } from '@/components/ui/toast-provider';
import { CommandPalette } from '@/components/ui/command-palette';
import { Sidebar } from '@/components/layout/sidebar';
import { Header } from '@/components/layout/header';

import { OverviewScreen } from '@/components/screens/overview-screen';
import { TrainingScreen } from '@/components/screens/training-screen';
import { ModelsScreen } from '@/components/screens/models-screen';
import { ChatScreen } from '@/components/screens/chat-screen';
import { DatasetsScreen } from '@/components/screens/datasets-screen';
import { DeploymentsScreen } from '@/components/screens/deployments-screen';

type Screen = 'overview' | 'training' | 'models' | 'chat' | 'datasets' | 'deployments';

const POLL_MS = 4000;

export default function DashboardPage() {
  const [loading, setLoading] = useState(true);
  const [screen, setScreen] = useState<Screen>('overview');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [cmdOpen, setCmdOpen] = useState(false);

  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<number | null>(null);
  const [runs, setRuns] = useState<Run[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [modelCatalog, setModelCatalog] = useState<ModelCatalogResponse | null>(null);

  const supportedModels = useMemo(() => modelCatalog?.supported_models ?? [], [modelCatalog]);
  const registeredModels = useMemo(() => modelCatalog?.registered_models ?? [], [modelCatalog]);
  const activeRunCount = useMemo(() => runs.filter(r => r.status === 'running' || r.status === 'pending').length, [runs]);

  // ⌘K global shortcut
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') { e.preventDefault(); setCmdOpen(v => !v); }
      if (e.key === 'Escape') setCmdOpen(false);
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  const refreshModelCatalog = useCallback(async () => {
    try { setModelCatalog(await getModelCatalog()); } catch { /* non-fatal */ }
  }, []);

  const refreshDatasets = useCallback(async () => {
    try { setDatasets(await getDatasets()); } catch { /* non-fatal */ }
  }, []);

  const refreshRuns = useCallback(async (projectId: number) => {
    try {
      const data = await getRuns(projectId);
      setRuns(data);
      setSelectedProjectId(prev => prev ?? projectId);
    } catch { /* non-fatal */ }
  }, []);

  useEffect(() => {
    const bootstrap = async () => {
      setLoading(true);
      try {
        let projectList = await getProjects();
        if (!projectList.length) {
          const p = await createProject({ name: 'My Project', description: 'Auto-created project.' });
          projectList = [p];
        }
        setProjects(projectList);
        const pid = projectList[0].id;
        setSelectedProjectId(pid);
        await Promise.allSettled([refreshRuns(pid), refreshModelCatalog(), refreshDatasets()]);
      } finally {
        setLoading(false);
      }
    };
    void bootstrap();
  }, [refreshDatasets, refreshModelCatalog, refreshRuns]);

  useEffect(() => {
    if (!selectedProjectId) return;
    const id = setInterval(() => void refreshRuns(selectedProjectId), POLL_MS);
    return () => clearInterval(id);
  }, [refreshRuns, selectedProjectId]);

  async function handleCreateProject() {
    const name = prompt('Project name:');
    if (!name?.trim()) return;
    try {
      const p = await createProject({ name: name.trim() });
      setProjects(prev => [p, ...prev]);
      setSelectedProjectId(p.id);
      void refreshRuns(p.id);
    } catch { /* swallow */ }
  }

  return (
    <ToastProvider>
      <div style={{ display: 'flex', height: '100vh', overflow: 'hidden', background: 'var(--bg)' }}>
        <Sidebar screen={screen} setScreen={setScreen} collapsed={sidebarCollapsed} onCmd={() => setCmdOpen(true)} activeRunCount={activeRunCount} />
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', minWidth: 0 }}>
          <Header
            collapsed={sidebarCollapsed}
            setCollapsed={setSidebarCollapsed}
            projects={projects}
            selectedProjectId={selectedProjectId}
            onSelectProject={id => { setSelectedProjectId(id); void refreshRuns(id); }}
            onCreateProject={handleCreateProject}
            onCmd={() => setCmdOpen(true)}
          />
          <main style={{ flex: 1, overflow: 'hidden' }}>
            {loading ? (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', flexDirection: 'column', gap: 16 }}>
                <div style={{ width: 44, height: 44, borderRadius: 12, background: 'linear-gradient(135deg,var(--acc),#8b5cf6)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <div style={{ width: 20, height: 20, borderRadius: 999, border: '2px solid rgba(255,255,255,.4)', borderTopColor: '#fff', animation: 'spin 1s linear infinite' }} />
                </div>
                <div style={{ fontSize: 14, color: 'var(--muted-text)' }}>Loading Tuner UI…</div>
              </div>
            ) : (
              <>
                {screen === 'overview' && <OverviewScreen projects={projects} runs={runs} datasets={datasets} setScreen={setScreen} />}
                {screen === 'training' && <TrainingScreen runs={runs} datasets={datasets} supportedModels={supportedModels} selectedProjectId={selectedProjectId} onRunCreated={run => setRuns(prev => [run, ...prev])} onRefreshRuns={() => { if (selectedProjectId) void refreshRuns(selectedProjectId); }} setScreen={setScreen} />}
                {screen === 'models' && <ModelsScreen supportedModels={supportedModels} registeredModels={registeredModels} setScreen={setScreen} />}
                {screen === 'chat' && <ChatScreen supportedModels={supportedModels} registeredModels={registeredModels} setScreen={setScreen} />}
                {screen === 'datasets' && <DatasetsScreen datasets={datasets} onDatasetCreated={d => setDatasets(prev => [d, ...prev])} setScreen={setScreen} />}
                {screen === 'deployments' && <DeploymentsScreen setScreen={setScreen} />}
              </>
            )}
          </main>
        </div>
      </div>
      <CommandPalette open={cmdOpen} onClose={() => setCmdOpen(false)} setScreen={setScreen} />
    </ToastProvider>
  );
}
