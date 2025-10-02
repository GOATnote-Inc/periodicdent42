import type { ReactNode } from "react";

type ResultPaneProps = {
  title: string;
  children: ReactNode;
  footer?: ReactNode;
};

export function ResultPane({ title, children, footer }: ResultPaneProps) {
  return (
    <section className="space-y-3 rounded-lg border bg-white p-4 shadow-sm">
      <header>
        <h2 className="text-base font-semibold">{title}</h2>
      </header>
      <div className="space-y-2 text-sm text-slate-700">{children}</div>
      {footer ? <footer className="border-t pt-3 text-xs text-muted-foreground">{footer}</footer> : null}
    </section>
  );
}
