import type { ReactNode } from "react";

type HowItWorksProps = {
  children: ReactNode;
};

export function HowItWorks({ children }: HowItWorksProps) {
  return (
    <section className="space-y-2 rounded-lg border bg-slate-50 p-4 text-sm text-slate-700">
      <h2 className="text-base font-semibold text-slate-900">How it works</h2>
      <div className="space-y-1 leading-relaxed">{children}</div>
    </section>
  );
}
