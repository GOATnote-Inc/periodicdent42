export default function HomePage() {
  return (
    <section className="space-y-4">
      <h2 className="text-lg font-semibold">Welcome</h2>
      <p className="text-sm text-slate-300">
        Use the navigation to inspect active runs and provenance graphs produced by the orchestrator.
      </p>
      <a
        className="inline-flex rounded bg-emerald-500 px-4 py-2 text-sm font-medium text-black"
        href="/runs"
      >
        View Runs
      </a>
    </section>
  );
}
