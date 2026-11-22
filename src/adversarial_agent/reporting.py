"""Report generation: HTML and Markdown for adversarial multi-agent findings."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict
import numpy as np


class _SafeEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and Python bools."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)

STYLE = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1.5rem; color: #1f2937; line-height: 1.6; }
h1, h2, h3 { color: #111827; }
.card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 1.2rem; margin: 1.5rem 0; background: #ffffff; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
.badge { display: inline-block; padding: 0.25rem 0.6rem; border-radius: 9999px; font-weight: 600; font-size: 0.85rem; }
.badge-robust { background: #dcfce7; color: #166534; }
.badge-simpson { background: #fef9c3; color: #854d0e; }
.badge-spurious { background: #fee2e2; color: #991b1b; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.95rem; }
th, td { border: 1px solid #e5e7eb; padding: 0.6rem 0.8rem; text-align: left; }
th { background: #f9fafb; font-weight: 600; }
code { background: #f3f4f6; padding: 0.15rem 0.4rem; border-radius: 4px; font-family: monospace; }
details { margin: 0.8rem 0; padding: 0.6rem; background: #f9fafb; border-radius: 6px; }
"""

def build_report(results: Dict[str, Any], out_path: Path) -> Dict[str, str]:
    out_path.mkdir(parents=True, exist_ok=True)
    html_file = out_path / "adversarial_audit_report.html"
    md_file = out_path / "adversarial_audit_report.md"

    html_file.write_text(_render_html(results), encoding="utf-8")
    md_file.write_text(_render_markdown(results), encoding="utf-8")
    (out_path / "findings.json").write_text(
        json.dumps(results["adversarial_findings"], indent=2, cls=_SafeEncoder), encoding="utf-8"
    )

    return {
        "html": str(html_file),
        "markdown": str(md_file),
        "json": str(out_path / "findings.json"),
    }

def _render_html(results: Dict[str, Any]) -> str:
    findings = results.get("adversarial_findings", [])
    profile = results.get("profile", {})
    
    sections = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Adversarial Data Analysis Audit Report</title>",
        f"<style>{STYLE}</style></head><body>",
        "<h1>⚔️ Adversarial Data Analysis Audit Report</h1>",
        f"<p>Input file: <code>{html.escape(str(results['input_file']))}</code> | "
        f"Rows: <strong>{profile.get('row_count', 'N/A')}</strong> | "
        f"Columns: <strong>{profile.get('column_count', 'N/A')}</strong></p>",
        "<hr/>",
        "<h2>Executive Summary</h2>",
        f"<p>The <strong>Explorer Agent</strong> evaluated candidate relationships in the dataset. "
        f"Each was stress-tested by the <strong>Devil's Advocate Agent</strong> for Simpson's Paradox reversals, "
        f"linear confounding attenuation, and bootstrap stability before final <strong>Referee</strong> adjudication.</p>",
    ]

    for f in findings:
        badge_cls = "badge-robust" if "ROBUST" in f["verdict_badge"] else ("badge-simpson" if "SIMPSON" in f["verdict_badge"] else "badge-spurious")
        sections.append("<div class='card'>")
        sections.append(f"<h3><span class='badge {badge_cls}'>{html.escape(f['verdict_badge'])}</span> {html.escape(f['claim']['source_variable'])} vs {html.escape(f['claim']['target_variable'])}</h3>")
        sections.append(f"<p><strong>🔍 Explorer Claim:</strong> {html.escape(f['claim']['description'])}</p>")
        sections.append(f"<p><strong>⚖️ Devil's Advocate Audit:</strong> {html.escape(f['audit']['falsification_reason'])}</p>")

        simpsons = f["audit"].get("simpsons_audit", {})
        if simpsons.get("cmh_odds_ratio") is not None:
            sections.append(f"<p><strong>📐 Cochran-Mantel-Haenszel (CMH) Test:</strong> Common Odds Ratio = <code>{simpsons['cmh_odds_ratio']:.3f}</code> (p = <code>{simpsons.get('cmh_p_value', 0.0):.3e}</code>)</p>")

        if simpsons.get("confounder_column") and simpsons.get("breakdowns"):
            sections.append(f"<details><summary><strong>Subgroup Breakdown across '{html.escape(simpsons['confounder_column'])}' (Click to view)</strong></summary><ul>")
            for b in simpsons["breakdowns"]:
                flag = " <span style='color: red; font-weight: bold;'>[REVERSED]</span>" if b["reverses_sign"] else ""
                sections.append(f"<li>{html.escape(b['description'])}{flag}</li>")
            sections.append("</ul></details>")

        sections.append(f"<p><strong>🏆 Referee Adjudication:</strong> {html.escape(f['synthesis_summary'])}</p>")
        
        rationale = f.get("classification_rationale", {})
        if rationale.get("why_assigned"):
            sections.append(f"<div style='background: #f8fafc; border-left: 4px solid #64748b; padding: 10px 14px; margin: 10px 0; border-radius: 4px;'>")
            sections.append(f"<p style='margin: 0 0 6px 0;'><strong>🎯 Why Categorized as <em>{html.escape(rationale.get('category_name', f['verdict_badge']))}</em>:</strong></p>")
            sections.append(f"<p style='margin: 0 0 6px 0; font-size: 0.95em;'>{html.escape(rationale['why_assigned'])}</p>")
            if rationale.get("decision_rule"):
                sections.append(f"<p style='margin: 0 0 4px 0; font-size: 0.85em; color: #475569;'><strong>Decision Rule:</strong> <code>{html.escape(rationale['decision_rule'])}</code></p>")
            if rationale.get("key_evidence"):
                sections.append("<ul style='margin: 4px 0 0 18px; font-size: 0.9em;'>")
                for ev in rationale["key_evidence"]:
                    sections.append(f"<li>{html.escape(ev)}</li>")
                sections.append("</ul>")
            sections.append("</div>")

        sections.append(f"<p><strong>💡 Actionable Takeaway:</strong> <em>{html.escape(f['actionable_takeaway'])}</em></p>")
        sections.append("</div>")

    sections.append("</body></html>")
    return "\n".join(sections)

def _render_markdown(results: Dict[str, Any]) -> str:
    findings = results.get("adversarial_findings", [])
    profile = results.get("profile", {})
    lines = [
        "# ⚔️ Adversarial Data Analysis Audit Report",
        "",
        f"- **Input File:** `{results['input_file']}`",
        f"- **Rows:** {profile.get('row_count', 'N/A')} | **Columns:** {profile.get('column_count', 'N/A')}",
        "",
        "---",
        "## Adjudicated Findings",
        "",
    ]
    for f in findings:
        lines.append(f"### {f['verdict_badge']}: {f['claim']['source_variable']} vs {f['claim']['target_variable']}")
        lines.append(f"- **Explorer Claim:** {f['claim']['description']}")
        lines.append(f"- **Devil's Advocate Audit:** {f['audit']['falsification_reason']}")
        simpsons = f["audit"].get("simpsons_audit", {})
        if simpsons.get("cmh_odds_ratio") is not None:
            lines.append(f"  - *CMH Stratified Common Odds Ratio:* `{simpsons['cmh_odds_ratio']:.3f}` (p = `{simpsons.get('cmh_p_value', 0.0):.3e}`)")
        if simpsons.get("confounder_column") and simpsons.get("breakdowns"):
            lines.append(f"  - *Subgroup breakdown across '{simpsons['confounder_column']}':*")
            for b in simpsons["breakdowns"]:
                flag = " **[REVERSED]**" if b["reverses_sign"] else ""
                lines.append(f"    - {b['description']}{flag}")
        lines.append(f"- **Referee Verdict:** {f['synthesis_summary']}")
        rationale = f.get("classification_rationale", {})
        if rationale.get("why_assigned"):
            lines.append(f"  - **🎯 Why Categorized as `{rationale.get('category_name', f['verdict_badge'])}`:**")
            lines.append(f"    - *Rationale:* {rationale['why_assigned']}")
            if rationale.get("decision_rule"):
                lines.append(f"    - *Rule:* `{rationale['decision_rule']}`")
            if rationale.get("key_evidence"):
                lines.append("    - *Key Evidence:*")
                for ev in rationale["key_evidence"]:
                    lines.append(f"      • {ev}")
        lines.append(f"- **Actionable Takeaway:** *{f['actionable_takeaway']}*")
        lines.append("")

    return "\n".join(lines)
