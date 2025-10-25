import React from "react";

type Guardrail = {
  name: string;
  passed: boolean;
  details?: string;
};

type GuardrailChipsProps = {
  guardrails: Guardrail[];
};

export const GuardrailChips: React.FC<GuardrailChipsProps> = ({ guardrails }) => {
  if (!guardrails.length) {
    return <span className="text-xs text-muted-foreground">No guardrail signals.</span>;
  }

  return (
    <div className="flex flex-wrap gap-2">
      {guardrails.map((guardrail) => {
        const color = guardrail.passed ? "bg-emerald-100 text-emerald-700" : "bg-rose-100 text-rose-700";
        return (
          <span key={guardrail.name} className={`rounded-full px-2 py-1 text-xs font-medium ${color}`}>
            {guardrail.name}: {guardrail.passed ? "pass" : "fail"}
          </span>
        );
      })}
    </div>
  );
};

export default GuardrailChips;
