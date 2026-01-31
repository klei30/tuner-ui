"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { CheckCircle, AlertCircle, Loader2, ExternalLink } from "lucide-react";
import { saveHFToken, removeHFToken, getHFStatus } from "@/lib/api";

export function HuggingFaceSettings() {
  const [token, setToken] = useState("");
  const [isConnected, setIsConnected] = useState(false);
  const [username, setUsername] = useState<string | null>(null);
  const [lastVerified, setLastVerified] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Check if token is already configured
  useEffect(() => {
    checkStatus();
  }, []);

  const checkStatus = async () => {
    setIsLoading(true);
    try {
      const data = await getHFStatus();
      setIsConnected(data.connected);
      setUsername(data.username || null);
      setLastVerified(data.last_verified || null);
    } catch (err) {
      console.error("Failed to check HF status:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSaveToken = async () => {
    if (!token.trim()) {
      setError("Please enter a HuggingFace token");
      return;
    }

    if (!token.startsWith("hf_")) {
      setError("Invalid token format. HuggingFace tokens start with 'hf_'");
      return;
    }

    setIsSaving(true);
    setError(null);
    setSuccess(null);

    try {
      const data = await saveHFToken(token);
      setIsConnected(true);
      setUsername(data.username || null);
      setLastVerified(data.last_verified || null);
      setToken(""); // Clear input
      setSuccess("HuggingFace token saved successfully!");
    } catch (err: any) {
      setError(err.message || "Failed to save token");
    } finally {
      setIsSaving(false);
    }
  };

  const handleRemoveToken = async () => {
    if (!confirm("Are you sure you want to remove your HuggingFace token?")) {
      return;
    }

    setIsSaving(true);
    setError(null);
    setSuccess(null);

    try {
      await removeHFToken();
      setIsConnected(false);
      setUsername(null);
      setLastVerified(null);
      setSuccess("HuggingFace token removed successfully");
    } catch (err: any) {
      setError(err.message || "Failed to remove token");
    } finally {
      setIsSaving(false);
    }
  };

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>HuggingFace Integration</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin" />
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>HuggingFace Integration</CardTitle>
        <CardDescription>
          Connect your HuggingFace account to enable one-click model deployment
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {isConnected ? (
            <div className="space-y-4">
              <Alert className="border-green-500 dark:border-green-700 bg-green-50 dark:bg-green-950/30">
                <CheckCircle className="h-4 w-4 text-green-600 dark:text-green-500" />
                <AlertDescription className="text-green-800 dark:text-green-300">
                  Connected as <strong>{username}</strong>
                  {lastVerified && (
                    <span className="text-sm block mt-1">
                      Last verified: {new Date(lastVerified).toLocaleString()}
                    </span>
                  )}
                </AlertDescription>
              </Alert>

              <div className="flex gap-2">
                <Button
                  variant="outline"
                  onClick={() => window.open("https://huggingface.co/" + username, "_blank")}
                >
                  <ExternalLink className="mr-2 h-4 w-4" />
                  View HuggingFace Profile
                </Button>
                <Button
                  variant="destructive"
                  onClick={handleRemoveToken}
                  disabled={isSaving}
                >
                  {isSaving ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Removing...
                    </>
                  ) : (
                    "Disconnect"
                  )}
                </Button>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <div>
                <Label htmlFor="hf-token">HuggingFace Access Token</Label>
                <Input
                  id="hf-token"
                  type="password"
                  placeholder="hf_..."
                  value={token}
                  onChange={(e) => setToken(e.target.value)}
                  className="mt-2"
                />
                <p className="text-sm text-muted-foreground mt-2">
                  Get your token from{" "}
                  <a
                    href="https://huggingface.co/settings/tokens"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="underline text-blue-600 hover:text-blue-800"
                  >
                    HuggingFace Settings
                  </a>
                  . Make sure to give it <strong>write</strong> permissions.
                </p>
              </div>

              <Button onClick={handleSaveToken} disabled={isSaving || !token.trim()}>
                {isSaving ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Connecting...
                  </>
                ) : (
                  "Connect HuggingFace"
                )}
              </Button>
            </div>
          )}

          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {success && (
            <Alert className="border-green-500 dark:border-green-700 bg-green-50 dark:bg-green-950/30">
              <CheckCircle className="h-4 w-4 text-green-600 dark:text-green-500" />
              <AlertDescription className="text-green-800 dark:text-green-300">{success}</AlertDescription>
            </Alert>
          )}

          <div className="rounded-lg bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-800 p-4 text-sm text-blue-800 dark:text-blue-300">
            <p className="font-semibold mb-2">What you can do with HuggingFace integration:</p>
            <ul className="list-disc list-inside space-y-1">
              <li>Deploy trained models with one click</li>
              <li>Automatic model card generation</li>
              <li>Share models publicly or privately</li>
              <li>Use HuggingFace Inference API</li>
            </ul>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
