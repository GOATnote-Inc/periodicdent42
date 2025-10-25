import type { ReactNode } from "react";

type ErrorBannerProps = {
  children: ReactNode;
  visible: boolean;
};

export function ErrorBanner({ children, visible }: ErrorBannerProps) {
  if (!visible) {
    return null;
  }
  return (
    <div className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">
      <strong className="font-semibold">Something went wrong:</strong>
      <div>{children}</div>
    </div>
  );
}
