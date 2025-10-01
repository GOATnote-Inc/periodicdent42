import Link from "next/link";

export default function HomePage() {
  return (
    <div className="card">
      <h2 className="text-xl font-semibold">Welcome</h2>
      <p className="mt-2 text-sm text-slate-300">
        Use the navigation above to select a partner, monitor metrics, capture feedback, and review pilot impact.
      </p>
      <div className="mt-4 flex gap-3 text-sm">
        <Link className="underline" href="/pilot">
          Start partner selection →
        </Link>
      </div>
    </div>
  );
}
