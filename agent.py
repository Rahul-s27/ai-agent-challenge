
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from subprocess import run
import sys
import argparse
import os
import importlib
from typing import Optional, TypedDict

from langgraph.graph import StateGraph, END

REPO_ROOT = Path(__file__).parent
DATA_DIR = REPO_ROOT / "data"
CUSTOM_DIR = REPO_ROOT / "custom_parsers"
MAX_ATTEMPTS = 3


class AgentState(TypedDict, total=False):
    bank: str
    pdf_path: Optional[str]
    csv_path: Optional[str]
    attempt: int
    success: bool
    error: Optional[str]
    parser_path: str


@dataclass
class Plan:
    bank: str
    pdf_path: Optional[Path]
    csv_path: Optional[Path]


def node_plan(state: AgentState) -> AgentState:
    bank = (state.get("bank") or "unknown").strip() if isinstance(state.get("bank"), str) else "unknown"
    pdf: Optional[Path] = None
    csv: Optional[Path] = None
    if bank != "unknown":
        d = DATA_DIR / bank
        if d.exists() and d.is_dir():
            pdf_candidate = d / f"{bank}_sample.pdf"
            csv_candidate = d / f"{bank}_sample.csv"
            pdf_fallback = d / f"{bank} sample.pdf"
            csv_fallback = d / "result.csv"
            if pdf_candidate.exists():
                pdf = pdf_candidate
            elif pdf_fallback.exists():
                pdf = pdf_fallback
            else:
                any_pdfs = sorted(d.glob("*.pdf"))
                pdf = any_pdfs[0] if any_pdfs else None
            if csv_candidate.exists():
                csv = csv_candidate
            elif csv_fallback.exists():
                csv = csv_fallback
            else:
                any_csvs = sorted(d.glob("*.csv"))
                csv = any_csvs[0] if any_csvs else None
    elif DATA_DIR.exists():
        for d in sorted(DATA_DIR.iterdir()):
            if d.is_dir():
                bank = d.name
                pdfs = sorted(list(d.glob("*sample.pdf")) or list(d.glob("*.pdf")))
                csvs = sorted(list(d.glob("*sample.csv")) or list(d.glob("*.csv")))
                pdf = pdfs[0] if pdfs else None
                csv = csvs[0] if csvs else None
                break
    return {
        "bank": bank,
        "pdf_path": str(pdf) if pdf else None,
        "csv_path": str(csv) if csv else None,
        "attempt": 0,
        "success": False,
        "error": None,
        "parser_path": str(CUSTOM_DIR / f"{bank}_parser.py"),
    }


def node_generate(state: AgentState) -> AgentState:
    CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
    state["attempt"] = int(state.get("attempt", 0)) + 1
    existing = Path(state.get("parser_path", ""))
    if existing.exists() and not state.get("error") and state["attempt"] == 1:
        return state
    csv_path = state.get("csv_path")
    src = generate_parser_with_gemini(
        bank=state.get("bank") or "unknown",
        csv_path=csv_path,
        error_context=state.get("error"),
    )
    Path(state["parser_path"]).write_text(src, encoding="utf-8")
    return state


def node_test(state: AgentState) -> AgentState:
    tests_dir = REPO_ROOT / "tests"
    try:
        importlib.import_module("pytest")
        has_pytest = True
    except Exception:
        has_pytest = False

    if has_pytest and tests_dir.exists():
        cp = run([sys.executable, "-m", "pytest", "-q"], cwd=str(REPO_ROOT), capture_output=True, text=True)
        state["success"] = cp.returncode == 0
        state["error"] = None if state["success"] else (cp.stdout + "\n" + cp.stderr)
    else:
        state["success"] = False
        state["error"] = "Tests unavailable (add tests/ and pytest)."
    return state


def node_self_fix(state: AgentState) -> AgentState:
    return state


def should_loop(state: AgentState) -> str:
    if state.get("success"):
        return END
    if state.get("attempt", 0) >= MAX_ATTEMPTS:
        return END
    return "generate"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("plan", node_plan)
    g.add_node("generate", node_generate)
    g.add_node("test", node_test)
    g.add_node("fix", node_self_fix)

    g.set_entry_point("plan")
    g.add_edge("plan", "generate")
    g.add_edge("generate", "test")
    g.add_edge("test", "fix")
    g.add_conditional_edges("fix", should_loop, {"generate": "generate", END: END})
    return g.compile()


def _load_env() -> None:
    """Lightweight .env loader to support GEMINI_API_KEY without extra deps."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                if k and (k not in os.environ):
                    os.environ[k] = v
    except Exception:
        pass


def generate_parser_with_gemini(bank: str, csv_path: Optional[str], error_context: Optional[str] = None) -> str:
    """Use Google Gemini to generate parser code. Falls back to a small scaffold on error."""
    _load_env()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return (
            "from __future__ import annotations\n"
            "import pandas as pd\n\n"
            "def parse(pdf_path: str) -> pd.DataFrame:\n"
            "    \"\"\"Return a DataFrame for the bank statement at pdf_path.\n"
            "    Contract: exact schema to be matched by tests (added later).\n"
            "    \"\"\"\n"
            "    return pd.DataFrame()\n"
        )

    try:
        import google.generativeai as genai
    except Exception:
        return (
            "from __future__ import annotations\n"
            "import pandas as pd\n\n"
            "def parse(pdf_path: str) -> pd.DataFrame:\n"
            "    return pd.DataFrame()\n"
        )

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        target_csv_note = f"The reference CSV (for schema) is at: {csv_path}." if csv_path else ""
        err_note = ("\nRecent test error:\n" + error_context.strip()) if error_context else ""
        prompt = (
            "You are generating a single Python module containing a bank statement parser.\n"
            "Constraints:\n"
            "- Deterministic output (temperature=0).\n"
            "- Only standard library + pandas.\n"
            "- File must define: `def parse(pdf_path: str) -> pandas.DataFrame`.\n"
            "- Parse the given PDF path using libraries such as camelot or pdfplumber and return\n"
            "  a pandas.DataFrame with the normalized transactions. Do NOT read the CSV file.\n"
            "  Use the CSV only as a schema reference if needed.\n"
            "- No extra prints, no main guard.\n\n"
            f"Bank: {bank}. {target_csv_note}{err_note}\n\n"
            "Output ONLY the Python code for the module, no explanations."
        )

        resp = model.generate_content(prompt, generation_config={"temperature": 0})
        text = (resp.text or "").strip()
        if text.startswith("```"):
            text = "\n".join(text.splitlines()[1:])
            if text.endswith("```"):
                text = "\n".join(text.splitlines()[:-1])
        return text if text else (
            "from __future__ import annotations\nimport pandas as pd\n\n"
            "def parse(pdf_path: str) -> pd.DataFrame:\n    return pd.DataFrame()\n"
        )
    except Exception:
        return (
            "from __future__ import annotations\nimport pandas as pd\n\n"
            "def parse(pdf_path: str) -> pd.DataFrame:\n    return pd.DataFrame()\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Agent for bank statement parsing")
    parser.add_argument("--target", required=True, help="Target bank code, e.g., icici")
    args = parser.parse_args()

    app = build_graph()
    final = app.invoke({"bank": args.target})
    success = bool(final.get("success"))
    print(f"success={success} attempts={final.get('attempt')} bank={final.get('bank')} parser={final.get('parser_path')}")
    if not success and final.get("error"):
        print("error:\n" + str(final["error"]))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
