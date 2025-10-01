const colors: Record<string, string> = {
  submitted: "bg-slate-700 text-slate-100",
  queued: "bg-blue-500 text-black",
  running: "bg-amber-400 text-black",
  completed: "bg-emerald-400 text-black",
  aborted: "bg-rose-500 text-white",
  error: "bg-rose-600 text-white"
};

export default function StatusBadge({ status }: { status: string }) {
  const color = colors[status.toLowerCase()] ?? "bg-slate-700 text-slate-100";
  return (
    <span className={`inline-flex items-center rounded px-3 py-1 text-xs font-semibold uppercase ${color}`}>
      {status}
    </span>
  );
}
