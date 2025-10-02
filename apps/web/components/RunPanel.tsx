"use client";

import { useState, type FormEvent } from "react";

type RunPanelProps = {
  title: string;
  description: string;
  placeholder?: string;
  onRun: (input: string) => Promise<void> | void;
  ctaLabel?: string;
};

export function RunPanel({ title, description, placeholder, onRun, ctaLabel = "Run" }: RunPanelProps) {
  const [value, setValue] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setIsSubmitting(true);
    try {
      await onRun(value);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-3 rounded-lg border bg-white p-4 shadow-sm">
      <div>
        <h2 className="text-base font-semibold">{title}</h2>
        <p className="text-sm text-muted-foreground">{description}</p>
      </div>
      <textarea
        value={value}
        onChange={(event) => setValue(event.target.value)}
        placeholder={placeholder}
        className="min-h-[120px] w-full rounded-md border px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
      />
      <button
        type="submit"
        className="inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-white transition hover:bg-primary/90 disabled:opacity-60"
        disabled={isSubmitting}
      >
        {isSubmitting ? "Running..." : ctaLabel}
      </button>
    </form>
  );
}
