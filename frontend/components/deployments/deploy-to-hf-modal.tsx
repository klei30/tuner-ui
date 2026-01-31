"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Loader2, Upload, AlertCircle, CheckCircle, ExternalLink } from "lucide-react";
import { deployToHuggingFace } from "@/lib/api";

interface DeployToHFModalProps {
  checkpointId: number;
  checkpointName: string;
  baseModel: string;
  runId: number;
}

export function DeployToHFModal({
  checkpointId,
  checkpointName,
  baseModel,
  runId,
}: DeployToHFModalProps) {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [repoName, setRepoName] = useState("");
  const [isPrivate, setIsPrivate] = useState(false);
  const [mergeWeights, setMergeWeights] = useState(true);
  const [isDeploying, setIsDeploying] = useState(false);
  const [deploymentStatus, setDeploymentStatus] = useState<
    "idle" | "success" | "error"
  >("idle");
  const [errorMessage, setErrorMessage] = useState("");
  const [repoUrl, setRepoUrl] = useState("");

  const handleDeploy = async () => {
    if (!repoName.trim()) {
      setErrorMessage("Please enter a repository name");
      setDeploymentStatus("error");
      return;
    }

    // Validate repo name format
    if (!repoName.includes("/")) {
      setErrorMessage("Repository name must be in format: username/model-name");
      setDeploymentStatus("error");
      return;
    }

    setIsDeploying(true);
    setDeploymentStatus("idle");
    setErrorMessage("");

    try {
      const data = await deployToHuggingFace(checkpointId, {
        repo_name: repoName,
        private: isPrivate,
        merge_weights: mergeWeights,
        create_inference_endpoint: false,
      });

      setRepoUrl(data.repo_url);
      setDeploymentStatus("success");

      // Close modal and redirect to deployments page after short delay
      setTimeout(() => {
        setOpen(false);
        router.push("/deployments");
      }, 1500); // 1.5 second delay to show success message
    } catch (error: any) {
      setErrorMessage(
        error.message || "Failed to deploy model. Please try again."
      );
      setDeploymentStatus("error");
    } finally {
      setIsDeploying(false);
    }
  };

  const handleClose = () => {
    setOpen(false);
    // Reset state after close animation
    setTimeout(() => {
      setDeploymentStatus("idle");
      setRepoName("");
      setErrorMessage("");
      setRepoUrl("");
    }, 300);
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="default" className="gap-2">
          <Upload className="h-4 w-4" />
          Deploy to HuggingFace
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[525px]">
        <DialogHeader>
          <DialogTitle>Deploy to HuggingFace Hub</DialogTitle>
          <DialogDescription>
            Upload your fine-tuned model to HuggingFace for sharing and
            inference.
          </DialogDescription>
        </DialogHeader>

        {deploymentStatus === "success" ? (
          <div className="space-y-4 py-4">
            <Alert className="border-green-500 dark:border-green-700 bg-green-50 dark:bg-green-950/30">
              <CheckCircle className="h-4 w-4 text-green-600 dark:text-green-500" />
              <AlertDescription className="text-green-800 dark:text-green-300">
                <div className="font-semibold mb-1">Deployment started successfully!</div>
                <div className="text-sm">Redirecting you to track progress...</div>
              </AlertDescription>
            </Alert>

            <div className="space-y-2">
              <Label>Repository URL</Label>
              <div className="flex gap-2">
                <Input value={repoUrl} readOnly className="font-mono text-sm" />
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => window.open(repoUrl, "_blank")}
                >
                  <ExternalLink className="h-4 w-4" />
                </Button>
              </div>
            </div>

            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>Opening deployments page...</span>
            </div>
          </div>
        ) : (
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="repo-name">
                Repository Name <span className="text-red-500">*</span>
              </Label>
              <Input
                id="repo-name"
                placeholder="username/my-fine-tuned-model"
                value={repoName}
                onChange={(e) => setRepoName(e.target.value)}
                disabled={isDeploying}
              />
              <p className="text-sm text-muted-foreground">
                Format: username/model-name or organization/model-name
              </p>
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="private">Private Repository</Label>
                  <p className="text-sm text-muted-foreground">
                    Only you can see this model
                  </p>
                </div>
                <Switch
                  id="private"
                  checked={isPrivate}
                  onCheckedChange={setIsPrivate}
                  disabled={isDeploying}
                />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label htmlFor="merge">Merge LoRA Weights</Label>
                  <p className="text-sm text-muted-foreground">
                    Create full model (recommended)
                  </p>
                </div>
                <Switch
                  id="merge"
                  checked={mergeWeights}
                  onCheckedChange={setMergeWeights}
                  disabled={isDeploying}
                />
              </div>
            </div>

            {deploymentStatus === "error" && errorMessage && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{errorMessage}</AlertDescription>
              </Alert>
            )}

            <div className="rounded-lg bg-muted/50 border border-border p-3">
              <p className="text-sm">
                <strong>Base Model:</strong> {baseModel}
                <br />
                <strong>Checkpoint:</strong> {checkpointName || `Step ${checkpointId}`}
                <br />
                <strong>Size:</strong> {mergeWeights ? "Full model" : "LoRA adapter only"}
              </p>
            </div>
          </div>
        )}

        <DialogFooter>
          {deploymentStatus === "success" ? (
            <Button onClick={handleClose}>Close</Button>
          ) : (
            <>
              <Button variant="outline" onClick={handleClose} disabled={isDeploying}>
                Cancel
              </Button>
              <Button
                onClick={handleDeploy}
                disabled={!repoName.trim() || isDeploying}
              >
                {isDeploying ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Deploying...
                  </>
                ) : (
                  <>
                    <Upload className="mr-2 h-4 w-4" />
                    Deploy
                  </>
                )}
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
