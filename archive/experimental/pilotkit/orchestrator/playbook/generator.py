from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict

import yaml

PLAYBOOK_TEMPLATE = """# Pilot Playbook: {pilot_name}

## Partner
- **Name:** {partner}

## Scope
{scope_block}

## Success Criteria
{success_block}

## Timeline
{timeline_block}

## Risks & Mitigations
{risks_block}

## Communications Plan
{comms_block}
"""

MOU_TEMPLATE = """# Memorandum of Understanding

This Memorandum of Understanding ("MOU") is entered into on {effective_date} between {company_name} ("Company") and {partner_name} ("Partner").

## Purpose
The parties agree to collaborate on the "{pilot_name}" pilot focused on {scope_summary}.

## Responsibilities
- Company provides the Intelligence Layer, workflow instrumentation, and support.
- Partner provides access to workflow data, subject matter experts, and change management support.

## Term
The pilot runs from {pilot_start} to {pilot_end} unless extended in writing.

## Confidentiality
Both parties agree to keep shared information confidential and use it solely for the pilot.

## Points of Contact
- Company: {company_contact}
- Partner: {partner_contact}

This MOU is non-binding and intended to memorialize the collaboration scope.
"""

DPA_TEMPLATE = """# Data Processing Addendum (Lightweight)

This Data Processing Addendum ("DPA") supplements the MOU between {company_name} and {partner_name} effective {effective_date}.

## Data Description
- Workflow metadata (timestamps, step names, unit identifiers) with PII redacted.
- Performance metrics derived from workflow instrumentation.

## Processing Instructions
- Company processes data solely to deliver pilot insights.
- Data is stored in secured internal systems with access logging.

## Security
- Encryption at rest and in transit.
- Access restricted to authorized pilot personnel.

## Data Subject Rights
Partner remains controller; Company will support reasonable data subject requests within 7 days.

## Deletion
All pilot data will be deleted or returned within 30 days of pilot completion unless otherwise agreed.
"""


def blockify(items) -> str:
    if isinstance(items, dict):
        return "\n".join([f"- **{k}:** {v}" for k, v in items.items()])
    if isinstance(items, list):
        return "\n".join([f"- {item}" for item in items])
    return str(items)


def generate_playbook(yaml_path: Path, output_dir: Path) -> Dict[str, Path]:
    data = yaml.safe_load(yaml_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    playbook_content = PLAYBOOK_TEMPLATE.format(
        pilot_name=data.get("pilot_name"),
        partner=data.get("partner"),
        scope_block=blockify(data.get("scope", {})),
        success_block=blockify(data.get("success_criteria", {})),
        timeline_block=blockify(data.get("timeline", {})),
        risks_block=blockify(data.get("risks", [])),
        comms_block=blockify(data.get("comms_plan", {})),
    )
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    playbook_path = output_dir / f"playbook-{timestamp}.md"
    playbook_path.write_text(playbook_content)

    legal_context = {
        "company_name": data.get("legal", {}).get("company_name", "Company"),
        "partner_name": data.get("legal", {}).get("partner_name", "Partner"),
        "effective_date": data.get("legal", {}).get("effective_date", "2025-01-01"),
        "pilot_name": data.get("pilot_name"),
        "scope_summary": data.get("scope", {}).get("workflow", "the pilot scope"),
        "pilot_start": data.get("timeline", {}).get("pilot_start", ""),
        "pilot_end": data.get("timeline", {}).get("pilot_end", ""),
        "company_contact": data.get("legal", {}).get("company_contact", "Pilot Lead"),
        "partner_contact": data.get("legal", {}).get("partner_contact", "Partner Lead"),
    }
    mou_path = output_dir / f"mou-{timestamp}.md"
    mou_path.write_text(MOU_TEMPLATE.format(**legal_context))
    dpa_path = output_dir / f"dpa-{timestamp}.md"
    dpa_path.write_text(DPA_TEMPLATE.format(**legal_context))

    return {"playbook": playbook_path, "mou": mou_path, "dpa": dpa_path}
