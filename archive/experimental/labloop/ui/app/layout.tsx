import "./globals.css";
import type { ReactNode } from "react";

export const metadata = {
  title: "LabLoop Provenance",
  description: "Internal provenance dashboard"
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-950 text-slate-100">
        <header className="border-b border-slate-800 bg-slate-900 p-4">
          <h1 className="text-xl font-semibold">LabLoop Provenance</h1>
          <nav className="mt-2 flex gap-4 text-sm text-slate-300">
            <a href="/runs">Runs</a>
          </nav>
        </header>
        <main className="p-6">{children}</main>
      </body>
    </html>
  );
}
