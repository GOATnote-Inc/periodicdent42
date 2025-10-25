import "./globals.css";
import Link from "next/link";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Pilot Intelligence Layer",
  description: "Pilot orchestration toolkit"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <div className="mx-auto flex max-w-6xl flex-col gap-6 p-6">
          <header className="flex items-center justify-between">
            <h1 className="text-2xl font-semibold">Intelligence Layer Pilot Kit</h1>
            <nav className="flex gap-4 text-sm text-slate-300">
              <Link href="/pilot">Pilot</Link>
              <Link href="/dashboard">Dashboard</Link>
              <Link href="/feedback">Feedback</Link>
              <Link href="/report/demo">Impact Report</Link>
            </nav>
          </header>
          <main className="grid gap-6">{children}</main>
        </div>
      </body>
    </html>
  );
}
