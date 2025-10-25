import "./globals.css";
import { ReactNode } from "react";

export const metadata = {
  title: "Synthesis Runs",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-slate-950 text-slate-100 min-h-screen">
        <header className="p-4 border-b border-slate-800">
          <h1 className="text-xl font-semibold">Synthesis Automation</h1>
        </header>
        <main className="p-6 space-y-6 max-w-5xl mx-auto">{children}</main>
      </body>
    </html>
  );
}
