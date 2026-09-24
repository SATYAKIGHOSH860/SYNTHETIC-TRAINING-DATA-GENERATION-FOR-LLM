"""Command-line entry point: a thin wrapper around src.pipeline.run_pipeline.

Examples:
    python main.py                              # 100 chunks (run.max_chunks)
    python main.py --max-chunks 5               # quick smoke test
    python main.py --resume                     # continue an interrupted run
    python main.py --refill                     # after a prompt upgrade: regenerate only incomplete chunks
    python main.py --min-quality-score 5 --similarity-threshold 0.9
    python main.py --set generate.questions_per_chunk=2
    python main.py --push-to-hub your-name/who-guidelines-qa

Why ``run.max_chunks: 100`` by default: the three guideline PDFs alone give
over a thousand chunks, hours of rate-limited API calls. Prove the pipeline
on a subset first; pass ``--max-chunks all`` deliberately.
"""

from __future__ import annotations

import argparse
import sys
import time


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a validated, deduplicated Q&A dataset from PDFs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--max-chunks", help="number of chunks to process, or 'all' (default: config run.max_chunks)")
    parser.add_argument("--min-quality-score", type=int, help="keep pairs scoring at least this (0-6)")
    parser.add_argument("--similarity-threshold", type=float, help="dedup cosine threshold (0-1)")
    parser.add_argument("--resume", action="store_true", help="resume from the last checkpoint of an identical run")
    parser.add_argument("--refill", action="store_true",
                        help="after a prompt upgrade: keep chunks that already gave the full set of pairs, "
                             "regenerate the rest with the current prompt (implies --resume)")
    parser.add_argument("--push-to-hub", metavar="REPO_ID", help="upload the final dataset to this HF dataset repo")
    parser.add_argument("--public", action="store_true", help="make the Hub dataset public (default private)")
    parser.add_argument("--pdf-dir", help="input folder (default: config paths.raw_dir)")
    parser.add_argument("--output-dir", help="output folder (default: config paths.output_dir)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override any config value, e.g. --set validate.min_quality_score=5")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()
    try:
        # Imported inside the try so a missing/invalid config.yaml is reported cleanly.
        from src.config import CONFIG, ConfigError, with_overrides
    except Exception as exc:  # ConfigError raised at import time
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    from src.export import load_jsonl, push_to_hub
    from src.pipeline import PipelineError, run_pipeline

    overrides = list(args.set)
    if args.min_quality_score is not None:
        overrides.append(f"validate.min_quality_score={args.min_quality_score}")
    if args.similarity_threshold is not None:
        overrides.append(f"deduplicate.similarity_threshold={args.similarity_threshold}")
    max_chunks = None
    if args.max_chunks is not None:
        if args.max_chunks.lower() in ("all", "none", "null"):
            overrides.append("run.max_chunks=null")
        else:
            try:
                max_chunks = int(args.max_chunks)
                if max_chunks <= 0:
                    raise ValueError
            except ValueError:
                print("ERROR: --max-chunks must be a positive integer or 'all'", file=sys.stderr)
                return 2
    try:
        cfg = with_overrides(CONFIG, overrides) if overrides else CONFIG
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    pdf_dir = args.pdf_dir or cfg.paths.raw_dir
    output_dir = args.output_dir or cfg.paths.output_dir
    print("=" * 60)
    print(" SYNTHETIC TRAINING DATA GENERATION")
    print("=" * 60)
    print(f" Generator model : {cfg.model.id}")
    print(f" Judge model     : {cfg.model.judge_id}")
    print(f" Chunk cap       : {max_chunks if max_chunks is not None else cfg.run.max_chunks or 'all'}")
    print(f" Quality cut-off : >= {cfg.validate.min_quality_score}/6")
    print(f" Dedup threshold : {cfg.deduplicate.similarity_threshold}")
    print(f" Resume          : {'refill (upgrade incomplete chunks)' if args.refill else 'yes' if args.resume else 'no'}")

    try:
        stats = run_pipeline(pdf_dir, output_dir, cfg, max_chunks=max_chunks, resume=args.resume,
                             refill=args.refill)
    except PipelineError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Finished chunks are checkpointed; run again with --resume to continue.",
              file=sys.stderr)
        return 130

    if args.push_to_hub:
        try:
            from src.config import resolve_path
            final = load_jsonl(resolve_path(output_dir) / "synthetic_dataset.jsonl")
            push_to_hub(final, args.push_to_hub, private=not args.public, stats=stats)
        except Exception as exc:
            print(f"\nERROR: upload to the Hub failed: {exc}", file=sys.stderr)
            return 1

    run = stats["run"]
    minutes, seconds = divmod(time.time() - started, 60)
    print(f"\nTotal wall time: {int(minutes)}m {seconds:.0f}s")
    print(f"API calls: {run['api_calls']} "
          f"(tokens: {run['tokens_generation']:,} generation + {run['tokens_validation']:,} validation)")
    print(f"Dataset: {output_dir}/synthetic_dataset.jsonl  |  Dashboard: streamlit run dashboard/app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
