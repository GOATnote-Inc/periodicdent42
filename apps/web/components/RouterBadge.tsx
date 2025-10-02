import React from "react";

type RouterBadgeProps = {
  arm: string;
  policy: string;
};

export const RouterBadge: React.FC<RouterBadgeProps> = ({ arm, policy }) => {
  return (
    <span className="inline-flex items-center gap-2 rounded-full border border-dashed px-3 py-1 text-xs font-medium">
      <span className="rounded-full bg-blue-100 px-2 py-0.5 text-blue-700">{arm}</span>
      <span className="text-muted-foreground">via {policy}</span>
    </span>
  );
};

export default RouterBadge;
