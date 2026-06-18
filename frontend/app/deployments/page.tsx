"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { useRouter } from "next/navigation";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { ArrowLeft, ExternalLink, Loader2, RefreshCw, Rocket, Package, Lock, Globe, CheckCircle2, XCircle, Clock } from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import { cn } from "@/lib/utils";
import { listDeployments } from "@/lib/api";

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

export default function DeploymentsPage() {
  const router = useRouter();
  const [deployments, setDeployments] = useState<Deployment[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const hasDataRef = useRef(false);

  const fetchDeployments = useCallback(async (retryCount = 0) => {
    console.log("[Deployments] Fetching...");

    try {
      const data = await listDeployments();
      console.log("Deployments data:", data);
      hasDataRef.current = true;
      setDeployments(data);
      setError(null);

      // Check if we need to keep polling
      const hasActiveDeployments = data.some(
        (d: Deployment) => d.status === 'pending' || d.status === 'uploading'
      );

      console.log("Has active deployments:", hasActiveDeployments);

      // Manage polling interval
      if (hasActiveDeployments && !intervalRef.current) {
        // Start polling if there are active deployments and we're not already polling
        console.log("Starting polling interval");
        intervalRef.current = setInterval(fetchDeployments, 5000);
      } else if (!hasActiveDeployments && intervalRef.current) {
        // Stop polling if no active deployments
        console.log("Stopping polling interval");
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    } catch (err: any) {
      console.error("[Deployments] Error:", err);

      // Only show error if we don't have deployments yet
      if (!hasDataRef.current) {
        setError(`Failed to connect to backend: ${err.message || "Network error"}. Make sure backend is running.`);
      } else {
        console.warn("[Deployments] Polling failed, will retry on next interval");
      }
    } finally {
      console.log("[Deployments] Setting isLoading to false");
      setIsLoading(false);
    }
  }, []);

  // Initial fetch and cleanup
  useEffect(() => {
    fetchDeployments();

    return () => {
      // Cleanup interval on unmount
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [fetchDeployments]);

  const getStatusBadge = (status: string) => {
    const variants: Record<string, { variant: any; label: string; icon: any; className: string }> = {
      completed: {
        variant: "default",
        label: "Completed",
        icon: CheckCircle2,
        className: "bg-green-100 text-green-700 dark:bg-green-500/10 dark:text-green-400 border-green-300 dark:border-green-500/20 hover:bg-green-200 dark:hover:bg-green-500/20"
      },
      pending: {
        variant: "secondary",
        label: "Pending",
        icon: Clock,
        className: "bg-yellow-100 text-yellow-700 dark:bg-yellow-500/10 dark:text-yellow-400 border-yellow-300 dark:border-yellow-500/20"
      },
      uploading: {
        variant: "secondary",
        label: "Uploading",
        icon: Rocket,
        className: "bg-blue-100 text-blue-700 dark:bg-blue-500/10 dark:text-blue-400 border-blue-300 dark:border-blue-500/20"
      },
      failed: {
        variant: "destructive",
        label: "Failed",
        icon: XCircle,
        className: "bg-red-100 text-red-700 dark:bg-red-500/10 dark:text-red-400 border-red-300 dark:border-red-500/20"
      },
    };

    const config = variants[status] || { variant: "secondary", label: status, icon: Clock, className: "" };
    const IconComponent = config.icon;

    return (
      <Badge variant={config.variant as any} className={cn("capitalize flex items-center gap-1.5 px-3 py-1 font-semibold", config.className)}>
        <IconComponent className="h-3.5 w-3.5" />
        {config.label}
      </Badge>
    );
  };

  const getProgressValue = (status: string, deployed_at: string | null) => {
    if (status === "completed") return 100;
    if (status === "uploading") return 75;
    if (status === "pending") return 25;
    return 0;
  };

  if (isLoading) {
    return (
      <div className="container py-8">
        <div className="mb-6">
          <h1 className="text-3xl font-bold">Deployments</h1>
          <p className="text-muted-foreground mt-1">
            Loading your deployments...
          </p>
        </div>
        <Card>
          <CardContent className="py-12">
            <div className="flex flex-col items-center justify-center space-y-4">
              <Loader2 className="h-12 w-12 animate-spin text-primary" />
              <div className="text-center space-y-1">
                <p className="text-sm font-medium">Connecting to backend...</p>
                <p className="text-xs text-muted-foreground">
                  If this takes more than 5 seconds, the backend may not be running
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container py-8">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => router.back()}
          className="mb-4"
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back
        </Button>
        <Card className="border-destructive">
          <CardHeader>
            <CardTitle className="text-destructive">Connection Error</CardTitle>
            <CardDescription>{error}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="rounded-lg bg-muted p-4 text-sm space-y-2">
              <p className="font-semibold">Troubleshooting steps:</p>
              <ol className="list-decimal list-inside space-y-1 text-muted-foreground">
                <li>Check if backend is running on <code className="bg-background px-1 py-0.5 rounded">http://127.0.0.1:8000</code></li>
                <li>Open browser console (F12) to see detailed error</li>
                <li>Restart backend: <code className="bg-background px-1 py-0.5 rounded">cd backend && uvicorn main:app --reload</code></li>
                <li>Check backend logs for errors</li>
              </ol>
            </div>
            <div className="flex gap-2">
              <Button onClick={() => fetchDeployments()} variant="outline">
                <RefreshCw className="mr-2 h-4 w-4" />
                Try Again
              </Button>
              <Button
                variant="secondary"
                onClick={() => window.open("http://127.0.0.1:8000/docs", "_blank")}
              >
                Open API Docs
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="container py-8 max-w-6xl mx-auto">
      <div className="mb-8">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => router.back()}
          className="mb-6 hover:bg-accent"
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back
        </Button>
        <div className="flex justify-between items-start">
          <div className="space-y-1">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-100 dark:bg-primary/10 rounded-lg">
                <Rocket className="h-6 w-6 text-blue-600 dark:text-primary" />
              </div>
              <h1 className="text-4xl font-bold tracking-tight">Deployments</h1>
            </div>
            <p className="text-slate-600 dark:text-muted-foreground text-lg ml-14">
              Track and manage your HuggingFace model deployments
            </p>
          </div>
          <Button onClick={() => fetchDeployments()} variant="outline" size="sm" className="gap-2 border-2">
            <RefreshCw className="h-4 w-4" />
            Refresh
          </Button>
        </div>
      </div>

      {deployments.length === 0 ? (
        <Card className="border-dashed border-2">
          <CardHeader className="text-center pb-4">
            <div className="mx-auto p-3 bg-slate-100 dark:bg-muted rounded-full w-fit mb-4">
              <Package className="h-8 w-8 text-slate-600 dark:text-muted-foreground" />
            </div>
            <CardTitle className="text-2xl">No Deployments Yet</CardTitle>
            <CardDescription className="text-base text-slate-600 dark:text-muted-foreground">
              Deploy your first model to HuggingFace Hub from the Runs page
            </CardDescription>
          </CardHeader>
          <CardContent className="text-center pb-6">
            <Button onClick={() => (window.location.href = "/")} size="lg" className="gap-2">
              <Rocket className="h-4 w-4" />
              Go to Runs
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-6">
          {deployments.map((deployment) => (
            <Card key={deployment.id} className="border-2 border-slate-200 dark:border-border hover:border-blue-300 dark:hover:border-primary/50 transition-colors shadow-sm">
              <CardHeader className="pb-4 bg-slate-50/50 dark:bg-transparent">
                <div className="flex justify-between items-start gap-4">
                  <div className="space-y-2 flex-1">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-blue-100 dark:bg-primary/10 rounded-md border border-blue-200 dark:border-transparent">
                        <Package className="h-5 w-5 text-blue-600 dark:text-primary" />
                      </div>
                      <CardTitle className="text-2xl font-bold">
                        {deployment.hf_repo_name}
                      </CardTitle>
                    </div>
                    <CardDescription className="text-base ml-11 text-slate-600 dark:text-muted-foreground">
                      {deployment.deployed_at
                        ? `Deployed ${formatDistanceToNow(new Date(deployment.deployed_at), { addSuffix: true })}`
                        : "Deployment in progress"}
                    </CardDescription>
                  </div>
                  {getStatusBadge(deployment.status)}
                </div>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Progress Bar - Always show for active deployments */}
                {(deployment.status === "pending" || deployment.status === "uploading" || deployment.status === "completed") && (
                  <div className="space-y-3 p-4 rounded-lg bg-slate-50 dark:bg-muted/50 border-2 border-slate-200 dark:border-border">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        {deployment.status === "completed" ? (
                          <CheckCircle2 className="h-5 w-5 text-green-600 dark:text-green-500" />
                        ) : (
                          <Loader2 className="h-5 w-5 animate-spin text-blue-600 dark:text-primary" />
                        )}
                        <div>
                          <p className="text-sm font-semibold text-foreground">
                            {deployment.status === "pending" && "Preparing deployment..."}
                            {deployment.status === "uploading" && "Uploading to HuggingFace Hub..."}
                            {deployment.status === "completed" && "Deployment completed successfully"}
                          </p>
                          <p className="text-xs text-slate-600 dark:text-muted-foreground mt-0.5">
                            {deployment.status === "pending" && "Downloading checkpoint and preparing files"}
                            {deployment.status === "uploading" && "Pushing model files to HuggingFace"}
                            {deployment.status === "completed" && "Model is live on HuggingFace Hub"}
                          </p>
                        </div>
                      </div>
                      <span className="text-sm font-bold text-blue-600 dark:text-primary">
                        {getProgressValue(deployment.status, deployment.deployed_at)}%
                      </span>
                    </div>
                    <Progress
                      value={getProgressValue(deployment.status, deployment.deployed_at)}
                      className="h-3"
                    />
                  </div>
                )}

                {/* Deployment Details */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="flex items-start gap-3 p-3 rounded-lg bg-slate-100 dark:bg-muted/30 border border-slate-200 dark:border-transparent">
                    <Package className="h-5 w-5 text-slate-600 dark:text-muted-foreground mt-0.5" />
                    <div>
                      <p className="text-xs text-slate-600 dark:text-muted-foreground font-semibold uppercase tracking-wide">Repository</p>
                      <p className="font-semibold text-sm mt-1.5 text-foreground">{deployment.hf_model_id}</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3 p-3 rounded-lg bg-slate-100 dark:bg-muted/30 border border-slate-200 dark:border-transparent">
                    {deployment.is_private ? (
                      <Lock className="h-5 w-5 text-slate-600 dark:text-muted-foreground mt-0.5" />
                    ) : (
                      <Globe className="h-5 w-5 text-slate-600 dark:text-muted-foreground mt-0.5" />
                    )}
                    <div>
                      <p className="text-xs text-slate-600 dark:text-muted-foreground font-semibold uppercase tracking-wide">Visibility</p>
                      <p className="font-semibold text-sm mt-1.5 text-foreground">
                        {deployment.is_private ? "Private" : "Public"}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3 p-3 rounded-lg bg-slate-100 dark:bg-muted/30 border border-slate-200 dark:border-transparent">
                    <Rocket className="h-5 w-5 text-slate-600 dark:text-muted-foreground mt-0.5" />
                    <div>
                      <p className="text-xs text-slate-600 dark:text-muted-foreground font-semibold uppercase tracking-wide">Model Type</p>
                      <p className="font-semibold text-sm mt-1.5 text-foreground">
                        {deployment.merged_weights ? "Full Model" : "LoRA Adapter"}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3 p-3 rounded-lg bg-slate-100 dark:bg-muted/30 border border-slate-200 dark:border-transparent">
                    <Package className="h-5 w-5 text-slate-600 dark:text-muted-foreground mt-0.5" />
                    <div>
                      <p className="text-xs text-slate-600 dark:text-muted-foreground font-semibold uppercase tracking-wide">Checkpoint ID</p>
                      <p className="font-semibold text-sm mt-1.5 text-foreground">#{deployment.checkpoint_id}</p>
                    </div>
                  </div>
                </div>

                {/* Error Message */}
                {deployment.error_message && (
                  <div className="rounded-lg bg-red-50 dark:bg-red-500/10 border-2 border-red-200 dark:border-red-500/20 p-4">
                    <div className="flex items-start gap-3">
                      <XCircle className="h-5 w-5 text-red-600 dark:text-red-500 mt-0.5 flex-shrink-0" />
                      <div>
                        <p className="text-sm font-semibold text-red-700 dark:text-red-400 mb-1">
                          Deployment Failed
                        </p>
                        <p className="text-sm text-red-600 dark:text-red-400/80">
                          {deployment.error_message}
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Actions */}
                <div className="flex gap-3 pt-2">
                  <Button
                    variant="default"
                    size="lg"
                    onClick={() => window.open(deployment.hf_repo_url, "_blank")}
                    disabled={deployment.status !== "completed"}
                    className="flex-1 gap-2"
                  >
                    <ExternalLink className="h-4 w-4" />
                    View on HuggingFace Hub
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
