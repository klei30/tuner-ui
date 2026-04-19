'use client';

import * as React from 'react';
import { Sidebar } from './sidebar';
import { Button } from '@/components/ui/button';
import { Menu, X } from 'lucide-react';
import { cn } from '@/lib/utils';

interface DashboardShellProps {
  children: React.ReactNode;
  sidebar?: React.ReactNode;
  topbar?: React.ReactNode;
}

export function DashboardShell({ children, sidebar, topbar }: DashboardShellProps) {
  const [sidebarOpen, setSidebarOpen] = React.useState(false);

  return (
    <div className="flex h-screen w-full overflow-hidden bg-background text-foreground">
      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-50 h-full w-64 shrink-0 border-r border-border bg-card/60 transition-transform duration-300 ease-in-out lg:static lg:translate-x-0",
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        <div className="flex h-16 items-center justify-between border-b border-border px-5 lg:hidden">
          <span className="text-lg font-semibold">Menu</span>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setSidebarOpen(false)}
          >
            <X className="h-5 w-5" />
          </Button>
        </div>
        {sidebar}
      </aside>

      <div className="flex h-full flex-1 flex-col overflow-hidden">
        {/* Mobile menu button in topbar */}
        <div className="lg:hidden">
          <div className="flex h-16 items-center border-b border-border bg-card/60 px-4">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setSidebarOpen(true)}
            >
              <Menu className="h-5 w-5" />
            </Button>
            <span className="ml-4 text-lg font-semibold">Tuner UI</span>
          </div>
        </div>

        <div className="border-b border-border bg-card/50 backdrop-blur-sm">
          {topbar}
        </div>
        <main className="flex-1 overflow-y-auto bg-background">
          <div className="mx-auto w-full max-w-7xl px-8 py-10 lg:px-12">{children}</div>
        </main>
      </div>
    </div>
  );
}
