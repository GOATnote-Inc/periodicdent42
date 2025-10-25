import "./globals.css";
import React from "react";

export const metadata = {
  title: "Mastery Demo",
  description: "Hybrid RAG mastery demo with synthetic data",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-50 text-slate-900">{children}</body>
    </html>
  );
}
