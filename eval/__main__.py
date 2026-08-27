"""CLI:  python -m eval <run|inspect|report|calibrate> [options]

    # check the layout first — no imports of boto3, no calls, no cost
    python -m eval doctor

    # see what the loader found before spending anything
    python -m eval inspect --dataset normad --n 40
    python -m eval inspect --dataset mdd --path data/MDD_100.csv

    # 20 stratified NormAd records, one cycle, hybrid classifier + control
    python -m eval run --dataset normad --n 20 --classifier hybrid --baseline

    # MDD, frames = passive/active agent (read from the row by the project)
    python -m eval run --dataset mdd --path data/MDD_100.csv --n 12

    # fit the free classifier's thresholds on the run you already paid for
    python -m eval calibrate --run-dir outputs/eval/normad-20-0824
    python -m eval report --run-dir outputs/eval/normad-20-0824 \\
        --thresholds outputs/eval/normad-20-0824/thresholds.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from . import calibrate as calib
from . import report as report_mod
from . import runner
from .config import EvalConfig, StructuralThresholds
from .datasets import LabelVocabulary, describe, load_records, sample_records


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", choices=["normad", "mdd"], default="normad")
    p.add_argument("--path", help="local dataset file (required for mdd)")
    p.add_argument("--split", default="train")
    p.add_argument("-n", "--n", type=int, default=20, help="how many records to run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-stratify", action="store_true",
                   help="sample at random instead of balancing gold labels")
    p.add_argument("--countries", nargs="*", default=[], help="normad filter")
    p.add_argument("--domains", nargs="*", default=[], help="mdd filter")
    p.add_argument("--normad-context", choices=["none", "country", "rot", "background"],
                   default="country")
    p.add_argument("--normad-challenger", default="neutral",
                   help="'neutral' leaves the challenger unframed; any other "
                        "string is used as its frame")
    p.add_argument("--mdd-label-field")
    p.add_argument("--mdd-scenario-field")
    p.add_argument("--src", help="ThirdSpace source root (else $THIRDSPACE_SRC / auto)")


def _cfg_from_args(args) -> EvalConfig:
    return EvalConfig(
        dataset=args.dataset,
        path=args.path,
        split=args.split,
        n=args.n,
        seed=args.seed,
        stratify=not args.no_stratify,
        countries=args.countries,
        domains=args.domains,
        normad_context=args.normad_context,
        normad_challenger=args.normad_challenger,
        mdd_label_field=args.mdd_label_field,
        mdd_scenario_field=args.mdd_scenario_field,
        src=args.src,
        max_cycles=getattr(args, "max_cycles", 1),
        convergence_metric=getattr(args, "convergence_metric", "embedding"),
        skip_generation=not getattr(args, "generate", False),
        mediator_frame=getattr(args, "mediator_frame", None),
        llm_call_budget=getattr(args, "budget", 400),
        workers=getattr(args, "workers", 1),
        classifier=getattr(args, "classifier", "hybrid"),
        thresholds=StructuralThresholds.from_file(getattr(args, "thresholds", None)),
        baseline=getattr(args, "baseline", False),
        readout_model=getattr(args, "readout_model", None),
        out_dir=getattr(args, "out_dir", None),
        run_id=getattr(args, "run_id", None),
        resume=getattr(args, "resume", False),
        persist_sessions=getattr(args, "persist_sessions", False),
    )


def cmd_doctor(args) -> int:
    """Verify the folder layout, imports and environment. Calls nothing."""
    from ._bootstrap import doctor

    checks = doctor(args.src, args.path)
    width = max(len(c["name"]) for c in checks)
    failed = 0
    for c in checks:
        mark = "PASS" if c["ok"] else "FAIL"
        if not c["ok"]:
            failed += 1
        print(f"[{mark}] {c['name']:<{width}}  {c['detail']}")
    print()
    if failed:
        print(f"{failed} check(s) failed. Expected layout:\n"
              "  <repo>/thirdspace/  models.py graph.py agents.py llm.py budget.py\n"
              "                      metrics.py main.py utils/{prompts,decorators,\n"
              "                      parsing,context}.py utils/decorator_registry/*.json\n"
              "  <repo>/eval/        this package\n"
              "  <repo>/.env         BEDROCK_MODEL_ID, CHALLENGER_MODEL_ID, ...")
        return 1
    print("All checks passed — `python -m eval run ...` is safe to start.")
    return 0


def cmd_inspect(args) -> int:
    """Load and sample, print what was found, run nothing."""
    cfg = _cfg_from_args(args)
    records = load_records(cfg)
    sample = sample_records(records, cfg.n, cfg.seed, cfg.stratify)
    vocab = LabelVocabulary.from_records(sample)

    print(f"\nloaded {len(records)} usable records; sampled {len(sample)}")
    print(json.dumps(describe(sample), indent=2))
    print("\nlabel vocabulary:")
    print(json.dumps(vocab.to_dict(), indent=2))
    if vocab.looks_free_text():
        print("\n!! That label column looks like free text, not a class. "
              "Pass --mdd-label-field <column>.")
    if sample:
        r = sample[0]
        print("\n--- first sampled record ---")
        print(f"id={r.record_id}  gold={r.gold}  group={r.group}")
        print(f"main frame     : {r.main_frame or '(neutral)'}")
        print(f"challenger     : {r.challenger_frame or '(neutral)'}")
        print(f"scenario ({len(r.scenario)} chars):\n{r.scenario[:900]}")
        if len(sample) > 1 and cfg.dataset == "mdd":
            print(f"\ncolumns present: {sorted(sample[0].raw)}")
    return 0


def cmd_run(args) -> int:
    cfg = _cfg_from_args(args)
    if cfg.run_id is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        cfg.run_id = f"{cfg.dataset}-n{cfg.n}-{cfg.classifier}-{stamp}"
    if cfg.dataset == "mdd" and not cfg.path:
        print("mdd needs --path to the MDD csv (e.g. code/MDD/MDD_100.csv)", file=sys.stderr)
        return 2

    result = runner.run(cfg)
    report = report_mod.build(result["rows"], result["vocab"],
                              config={"config": cfg.to_dict()}, seed=cfg.seed)
    md_path, json_path = report_mod.write(report, result["run_dir"])
    print(report_mod.to_markdown(report))
    print(f"\nwritten: {md_path}\n         {json_path}\n"
          f"         {result['run_dir'] / 'predictions.jsonl'}")
    return 0


def cmd_report(args) -> int:
    """Re-render (and optionally re-score) an existing run. No LLM calls."""
    run_dir = Path(args.run_dir)
    rows, vocab = calib.load_rows(run_dir)
    meta = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))

    if args.thresholds:
        # Re-run the structural head with new cut points and, when the run was
        # structural-only, re-derive the headline prediction from it.
        from .classifiers import classify_structural
        thresholds = StructuralThresholds.from_file(args.thresholds)
        mode = meta.get("config", {}).get("classifier")
        for row in rows:
            if not row.get("signals"):
                continue
            pred = classify_structural(row["signals"], vocab, thresholds)
            row["structural"] = pred.to_dict()
            if mode == "structural":
                row["prediction"] = pred.label
                row["correct"] = (pred.label == row.get("gold")) if pred.label else None
        meta.setdefault("config", {})["thresholds"] = thresholds.to_dict()

    report = report_mod.build(rows, vocab, config=meta, seed=args.seed)
    md_path, json_path = report_mod.write(report, run_dir)
    print(report_mod.to_markdown(report))
    print(f"\nwritten: {md_path}\n         {json_path}")
    return 0


def cmd_calibrate(args) -> int:
    run_dir = Path(args.run_dir)
    rows, vocab = calib.load_rows(run_dir)
    result = calib.calibrate(rows, vocab, folds=args.folds, seed=args.seed)
    out = run_dir / "thresholds.json"
    out.write_text(json.dumps(result["thresholds"], indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nwritten: {out}\nRe-score with:\n"
          f"  python -m eval report --run-dir {run_dir} --thresholds {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser(
        "doctor", help="verify layout, imports and env vars; run nothing")
    p_doctor.add_argument("--src", help="path to the thirdspace/ algorithm folder")
    p_doctor.add_argument("--path", help="optionally check a dataset file too")
    p_doctor.set_defaults(func=cmd_doctor)

    p_inspect = sub.add_parser("inspect", help="load + sample, print, run nothing")
    _add_common(p_inspect)
    p_inspect.set_defaults(func=cmd_inspect)

    p_run = sub.add_parser("run", help="run the pipeline and score it")
    _add_common(p_run)
    p_run.add_argument("--max-cycles", type=int, default=1)
    p_run.add_argument("--convergence-metric", choices=["embedding", "jaccard"],
                       default="embedding")
    p_run.add_argument("--generate", action="store_true",
                       help="also write the final document (costs one more call "
                            "per record; the read-out then sees it)")
    p_run.add_argument("--mediator-frame")
    p_run.add_argument("--budget", type=int, default=400,
                       help="LLM call budget per scenario")
    p_run.add_argument("--workers", type=int, default=1,
                       help="scenarios in parallel; cells already fan out 8x inside one")
    p_run.add_argument("--classifier", choices=["structural", "readout", "hybrid"],
                       default="hybrid")
    p_run.add_argument("--thresholds", help="json file from `eval calibrate`")
    p_run.add_argument("--baseline", action="store_true",
                       help="also predict with a single raw-record call (control)")
    p_run.add_argument("--readout-model", help="Bedrock model id for the classifier calls")
    p_run.add_argument("--out-dir", default=None,
                   help="default: <repo>/outputs/eval")
    p_run.add_argument("--run-id")
    p_run.add_argument("--resume", action="store_true")
    p_run.add_argument("--persist-sessions", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_report = sub.add_parser("report", help="re-render a finished run")
    p_report.add_argument("--run-dir", required=True)
    p_report.add_argument("--thresholds")
    p_report.add_argument("--seed", type=int, default=0)
    p_report.set_defaults(func=cmd_report)

    p_cal = sub.add_parser("calibrate", help="fit structural thresholds offline")
    p_cal.add_argument("--run-dir", required=True)
    p_cal.add_argument("--folds", type=int, default=5)
    p_cal.add_argument("--seed", type=int, default=0)
    p_cal.set_defaults(func=cmd_calibrate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
