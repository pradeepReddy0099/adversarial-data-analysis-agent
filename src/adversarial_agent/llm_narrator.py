"""LLM Narrator & Executive Copilot for Adversarial Data Analysis Agent.

Acts as a neuro-symbolic bridge: takes the deterministic Referee Dossier
and generates human-readable, domain-aware executive briefings and answers
user queries. Operates with offline heuristic fallback if no API key is provided.
"""
from __future__ import annotations

import os
import json
from typing import Dict, Any, Optional, List


class LLMNarrator:
    """Narrates and explains adjudicated findings using LLM APIs or offline heuristic synthesis."""

    def __init__(self, provider: str = "auto", api_key: Optional[str] = None):
        """
        Args:
            provider: 'auto', 'gemini', 'openai', 'anthropic', or 'offline'
            api_key: Optional explicit API key. If omitted, checks environment variables.
        """
        self.provider = provider
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.active_provider = self._detect_provider()

    def _detect_provider(self) -> str:
        if self.provider == "offline":
            return "offline"
        if os.environ.get("GEMINI_API_KEY") or (self.provider == "gemini" and self.api_key):
            return "gemini"
        if os.environ.get("OPENAI_API_KEY") or (self.provider == "openai" and self.api_key):
            return "openai"
        return "offline"

    def generate_executive_briefing(self, findings: List[Dict[str, Any]], dataset_name: str = "Dataset") -> str:
        """Generates an executive briefing explaining paradoxes, robust signals, and confounded relationships."""
        if self.active_provider == "gemini":
            try:
                return self._call_gemini(findings, dataset_name)
            except Exception as e:
                return self._offline_briefing(findings, dataset_name, error_note=f"Gemini API fallback: {e}")
        elif self.active_provider == "openai":
            try:
                return self._call_openai(findings, dataset_name)
            except Exception as e:
                return self._offline_briefing(findings, dataset_name, error_note=f"OpenAI API fallback: {e}")
        else:
            return self._offline_briefing(findings, dataset_name)

    def answer_query(self, query: str, findings: List[Dict[str, Any]], dataset_name: str = "Dataset") -> str:
        """Answers an interactive user question regarding the findings dossier."""
        context_str = json.dumps([{
            "pair": f"{f.get('claim', {}).get('source_variable')} -> {f.get('claim', {}).get('target_variable')}",
            "verdict": f.get("verdict"),
            "referee_summary": f.get("referee_explanation"),
            "takeaway": f.get("actionable_takeaway")
        } for f in findings], indent=2)

        prompt = (
            f"You are the Adversarial Data Analysis AI Copilot. The user is asking about an empirical audit of '{dataset_name}'.\n"
            f"Here is the adjudicated findings dossier:\n{context_str}\n\n"
            f"User Question: {query}\n\n"
            "Answer directly, citing exact subgroup reversals or confounders when relevant. Keep your answer concise (2-4 sentences)."
        )

        if self.active_provider == "gemini":
            try:
                import urllib.request
                key = self.api_key or os.environ.get("GEMINI_API_KEY")
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
                req = urllib.request.Request(
                    url,
                    data=json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            except Exception:
                pass

        # Offline fallback query answering
        query_lower = query.lower()
        matched = []
        for f in findings:
            src = f.get('claim', {}).get('source_variable', '').lower()
            tgt = f.get('claim', {}).get('target_variable', '').lower()
            if src in query_lower or tgt in query_lower or f.get('verdict', '').lower() in query_lower:
                matched.append(f)

        if matched:
            top = matched[0]
            return f"Regarding '{top.get('claim', {}).get('source_variable')}': {top.get('referee_explanation')} {top.get('actionable_takeaway')}"
        return f"Across {len(findings)} evaluated variables in '{dataset_name}', the Referee classified findings into: " + ", ".join(f"{f.get('claim', {}).get('source_variable')} ({f.get('verdict')})" for f in findings[:3])

    def _call_gemini(self, findings: List[Dict[str, Any]], dataset_name: str) -> str:
        import urllib.request
        key = self.api_key or os.environ.get("GEMINI_API_KEY")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
        
        prompt = (
            f"You are a Senior Principal Data Scientist and AI Auditor. Review these adjudicated findings from an adversarial multi-agent audit of '{dataset_name}'.\n"
            "Explain in clear, compelling, professional language:\n"
            "1. Why the detected Simpson's Paradoxes or confounded variables would mislead an executive who only looked at naive correlations.\n"
            "2. Which findings are verified as 100% robust and safe to make business or policy decisions upon.\n"
            "3. Immediate actionable recommendations for leadership.\n\n"
            f"Findings Data:\n{json.dumps(findings, indent=2)}\n\n"
            "Keep the response professional, well-structured with bullet points, and under 300 words."
        )

        payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    def _call_openai(self, findings: List[Dict[str, Any]], dataset_name: str) -> str:
        import urllib.request
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        url = "https://api.openai.com/v1/chat/completions"
        prompt = f"Analyze these audited statistical findings for '{dataset_name}' and provide a concise executive summary highlighting Simpson's Paradoxes and robust signals:\n{json.dumps(findings, indent=2)}"
        payload = json.dumps({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 400
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}"
        })
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()

    def _offline_briefing(self, findings: List[Dict[str, Any]], dataset_name: str, error_note: Optional[str] = None) -> str:
        """Deterministic heuristic narrative generator (Zero-API, 100% offline)."""
        robust = [f for f in findings if "ROBUST" in f.get("verdict", "")]
        paradox = [f for f in findings if "SIMPSON" in f.get("verdict", "")]
        confounded = [f for f in findings if "CONFOUNDED" in f.get("verdict", "") or "SPURIOUS" in f.get("verdict", "")]

        lines = [
            f"### 📋 Executive Briefing: Adversarial Audit of {dataset_name}",
            "",
            f"**Audit Summary:** Evaluated {len(findings)} candidate empirical claims through our dialectical multi-agent falsification engine."
        ]

        if paradox:
            lines.append("\n⚠️ **CRITICAL WARNING — Simpson's Paradox Detected:**")
            for p in paradox:
                var = p.get('claim', {}).get('source_variable', 'Variable')
                lines.append(f"- **{var}:** The apparent aggregate trend completely reverses across subgroups. *Action:* Do NOT make global policy decisions on {var}; interventions must be tailored at the sub-population level.")

        if confounded:
            lines.append("\n🔍 **Confounded / Spurious Associations Flagged:**")
            for c in confounded:
                var = c.get('claim', {}).get('source_variable', 'Variable')
                lines.append(f"- **{var}:** Relationship collapses when adjusting for covariates. The effect is mediated by lurking factors.")

        if robust:
            lines.append("\n✅ **Robust & Actionable Insights:**")
            for r in robust:
                var = r.get('claim', {}).get('source_variable', 'Variable')
                lines.append(f"- **{var}:** Withstood all four falsification batteries. Consistent across all sub-populations and statistically verified.")

        if error_note:
            lines.append(f"\n*(Running in local deterministic mode. {error_note})*")
        else:
            lines.append("\n*(Generated by ADAA Neuro-Symbolic Narrative Engine — set GEMINI_API_KEY or OPENAI_API_KEY to enable live LLM generation)*")

        return "\n".join(lines)
