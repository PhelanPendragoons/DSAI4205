"""Run Kai sentiment experiment modes locally (CUDA + DeepSeek teacher).

Examples (from any cwd):
  python models/kai/run_local.py --mode rationale-distillation
  python models/kai/run_local.py --mode counterfactual --distillation-export path/to/export.zip
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import runpy
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]  # group project


class _FakeIPython:
    def run_line_magic(self, name, arg):
        print(f"[run_local] skip %{name} {arg}", flush=True)


def get_ipython():  # noqa: N802 — notebook install cell expects this name
    return _FakeIPython()


def _load_env():
    os.environ.setdefault("KAI_ENV_FILE", str(HERE / ".env"))
    os.environ.setdefault("KAI_RUNS_ROOT", str(PROJECT / "runs" / "kai"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_DISABLED", "true")
    text = (HERE / ".env").read_text(encoding="utf-8").strip() if (HERE / ".env").is_file() else ""
    if not text:
        return
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if len(lines) == 1 and "=" not in lines[0]:
        os.environ.setdefault("TEACHER_API_KEY", lines[0].strip().strip('"').strip("'"))
        return
    for ln in lines:
        if "=" not in ln:
            continue
        key, _, value = ln.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _cell_source(cell) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return src


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=["label-only", "rationale-distillation", "counterfactual",
                                 "rationale-matched", "counterfactual-matched"])
    parser.add_argument("--distillation-export", default="",
                        help="Stage-2: path to stage-1 ZIP or extracted folder")
    parser.add_argument("--prior-export", default="", help="Matched arms: previous run ZIP or directory")
    parser.add_argument("--match-seed", type=int, default=42, help="Matched arms: seed for sampling training rows")
    parser.add_argument("--skip-install", action="store_true", default=True)
    parser.add_argument("--install", dest="skip_install", action="store_false")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the unified notebook first")
    parser.add_argument("--resume-dir", default="",
                        help="Reuse an existing RUN_DIR (skip finished experiment summaries)")
    args = parser.parse_args(argv)

    _load_env()
    if not (os.environ.get("TEACHER_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")):
        if args.mode in {"rationale-distillation", "counterfactual"}:
            raise SystemExit("缺少 TEACHER_API_KEY / DEEPSEEK_API_KEY（检查 models/kai/.env）")

    build = runpy.run_path(str(HERE / "build_notebook.py"))
    if args.rebuild:
        build["build"]()

    nb_path = PROJECT / "notebooks" / "kai" / "Kai_Sentiment_Experiments.ipynb"
    if not nb_path.is_file() or args.rebuild:
        build["build"]()
    notebook = json.loads(nb_path.read_text(encoding="utf-8"))

    ns = {
        "__name__": "__main__",
        "__file__": str(nb_path),
        "get_ipython": get_ipython,
    }
    # Make display() work without a notebook frontend.
    try:
        from IPython.display import display as _display
        ns["display"] = _display
    except Exception:
        ns["display"] = print

    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        cid = cell.get("id", "")
        source = _cell_source(cell)
        if not source.strip():
            continue
        if cid == "install" and args.skip_install:
            print("[run_local] skip install", flush=True)
            continue
        if cid == "mode":
            source = source.replace('RUN_MODE = "label-only"', f"RUN_MODE = {args.mode!r}", 1)
        if cid == "config" and args.mode.endswith("-matched"):
            path = args.prior_export or args.distillation_export
            if path:
                source = source.replace('PRIOR_EXPORT = ""', f"PRIOR_EXPORT = {str(Path(path).expanduser().resolve())!r}", 1)
            source = source.replace("MATCH_SEED = 42", f"MATCH_SEED = {args.match_seed}", 1)
        if cid == "config" and args.distillation_export:
            export = str(Path(args.distillation_export).expanduser().resolve())
            if 'DISTILLATION_EXPORT = ""' not in source:
                raise RuntimeError("config cell missing DISTILLATION_EXPORT placeholder")
            source = source.replace('DISTILLATION_EXPORT = ""', f"DISTILLATION_EXPORT = {export!r}", 1)
        if cid == "config" and args.resume_dir:
            resume = Path(args.resume_dir).expanduser().resolve()
            if not resume.is_dir():
                raise SystemExit(f"resume-dir 不存在: {resume}")
            marker = "RUN_ID = datetime.now(timezone.utc).strftime(\"%Y%m%dT%H%M%S_%fZ\") + f\"_seed{SEED}\""
            if marker not in source:
                raise RuntimeError("config cell missing RUN_ID assignment")
            source = source.replace(
                marker,
                f"RUN_ID = {resume.name!r}\n"
                f"OUTPUT_ROOT = Path({str(resume.parent)!r})\n"
                f"_RESUME_DIR = Path({str(resume)!r})",
                1,
            )
            source = source.replace(
                "RUN_DIR = OUTPUT_ROOT / RUN_ID\nRUN_DIR.mkdir(parents=True, exist_ok=False)",
                "RUN_DIR = _RESUME_DIR\nRUN_DIR.mkdir(parents=True, exist_ok=True)\n"
                "print(\"续跑目录:\", RUN_DIR)",
                1,
            )

        print(f"[run_local] RUN {cid}", flush=True)
        try:
            ast.parse(source)
            exec(compile(source, f"<{cid}>", "exec"), ns, ns)
        except Exception:
            traceback.print_exc()
            raise SystemExit(f"failed at cell {cid!r}")

    archive = ns.get("archive")
    run_dir = ns.get("RUN_DIR")
    print("[run_local] done", flush=True)
    print("RUN_DIR:", run_dir, flush=True)
    if archive:
        print("EXPORT:", archive, flush=True)
    summary_path = Path(os.environ["KAI_RUNS_ROOT"]) / f"last_{args.mode.replace('-', '_')}.json"
    payload = {
        "mode": args.mode,
        "run_dir": str(run_dir) if run_dir else None,
        "archive": str(archive) if archive else None,
        "teacher_model": ns.get("TEACHER_MODEL"),
        "teacher_base_url": ns.get("TEACHER_BASE_URL"),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY:", summary_path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
