import "../styles/globals.css";
import type { Metadata } from "next";
import { ReactNode } from "react";

export const metadata: Metadata = {
  title: "Midupdate Campaigns",
  description: "Monitor glass-box planner campaigns",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-slate-950 text-slate-50 min-h-screen">
        <div className="max-w-6xl mx-auto py-8 px-6">
          <header className="mb-10">
            <h1 className="text-3xl font-semibold">North-Star Campaigns</h1>
            <p className="text-slate-300">
              Live telemetry from the glass-box planner, including schema rationale snapshots and
              performance metrics.
            </p>
          </header>
          {children}
        </div>
      </body>
    </html>
  );
}
