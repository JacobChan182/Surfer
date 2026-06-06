#!/usr/bin/env python3
"""Repeatedly run Surfer simulations from random Wikipedia start pages."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from surfer.env import load_env
from surfer.agent import AgentConfig, run_agent
from surfer.llm_critic import LLMCritic
from surfer.nvidia_llm import DEFAULT_MODEL, NvidiaLLM, NvidiaLLMError
from surfer.weights_store import DEFAULT_WEIGHTS_PATH, load_weights
from surfer.wiki import WikiClient, WikiRateLimitError


def _log_line(log_path: Path | None, record: dict) -> None:
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def main() -> None:
    load_env()

    parser = argparse.ArgumentParser(
        description="Run repeated Surfer simulations from random Wikipedia pages (LLM + distillation)."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=0,
        help="Number of simulations (0 = run until Ctrl+C, default: 0)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Max steps per simulation (default: 200)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress per-step agent output (verbose is on by default)",
    )
    parser.add_argument(
        "--pick",
        choices=("bandit", "llm"),
        default="llm",
        help="Link selection: llm (default) or bandit (switch back after training)",
    )
    parser.add_argument(
        "--llm-candidate-limit",
        type=int,
        default=0,
        help="Max links sent to LLM picker (0 = all links, default: 0)",
    )
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_MODEL,
        help=f"NVIDIA NIM model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--llm-blend",
        type=float,
        default=0.6,
        help="LLM vs keyword reward blend (default: 0.6)",
    )
    parser.add_argument(
        "--llm-delta-scale",
        type=float,
        default=0.5,
        help="Scale for LLM weight deltas (default: 0.5)",
    )
    parser.add_argument(
        "--weights-file",
        type=Path,
        default=DEFAULT_WEIGHTS_PATH,
        help=f"Distilled weights file (default: {DEFAULT_WEIGHTS_PATH})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between simulations (default: 2.0)",
    )
    parser.add_argument(
        "--log",
        type=Path,
        metavar="PATH",
        help="Append one JSON line per simulation to this file",
    )
    args = parser.parse_args()
    verbose = not args.quiet

    try:
        llm = NvidiaLLM(model=args.llm_model)
    except NvidiaLLMError as exc:
        print(f"LLM setup failed: {exc}", file=sys.stderr)
        sys.exit(1)

    critic = LLMCritic(
        llm=llm,
        llm_blend=args.llm_blend,
        delta_scale=args.llm_delta_scale,
        candidate_limit=args.llm_candidate_limit,
        verbose=verbose,
    )

    total = 0
    successes = 0

    print("Starting simulation loop (Ctrl+C to stop)")
    print(f"Weights file: {args.weights_file}")
    print(f"LLM model: {args.llm_model}")
    print(f"Pick mode: {args.pick}")
    print()

    try:
        with WikiClient() as wiki:
            run_num = 0
            while args.runs == 0 or run_num < args.runs:
                run_num += 1
                total += 1

                try:
                    start_url = wiki.random_page_url()
                except WikiRateLimitError as exc:
                    print(f"Run {run_num}: Wikipedia rate limited fetching random page: {exc}")
                    time.sleep(args.delay * 5)
                    total -= 1
                    run_num -= 1
                    continue

                stored = load_weights(args.weights_file)
                stats = stored.stats if stored else None
                prior_runs = stats.runs if stats else 0
                prior_successes = stats.successes if stats else 0

                print(f"=== Run {run_num} | {start_url} ===")
                if stored and stored.weights and not verbose:
                    top = sorted(stored.weights.items(), key=lambda kv: kv[1], reverse=True)[:3]
                    print(f"  Loaded weights: {', '.join(f'{w}={v:.2f}' for w, v in top)}")

                config = AgentConfig(
                    max_steps=args.max_steps,
                    verbose=verbose,
                    use_llm=True,
                    pick_mode=args.pick,
                    llm_candidate_limit=args.llm_candidate_limit,
                    llm_blend=args.llm_blend,
                    llm_delta_scale=args.llm_delta_scale,
                    weights_file=args.weights_file,
                    save_weights=True,
                    weights_store=stored,
                )

                try:
                    result = run_agent(
                        start_url=start_url,
                        wiki=wiki,
                        critic=critic,
                        config=config,
                    )
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    print(f"Run {run_num} error: {exc}")
                    _log_line(
                        args.log,
                        {
                            "run": run_num,
                            "start_url": start_url,
                            "error": str(exc),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    time.sleep(args.delay)
                    continue

                if result.success:
                    successes += 1
                    print(f"SUCCESS in {result.steps} steps")
                else:
                    print(f"FAILED after {result.steps} steps: {result.reason}")

                updated = load_weights(args.weights_file)
                run_count = updated.stats.runs if updated else prior_runs + 1
                success_count = updated.stats.successes if updated else prior_successes

                print(
                    f"  Session: {successes}/{total} succeeded | "
                    f"All-time: {success_count}/{run_count} succeeded"
                )
                print()

                _log_line(
                    args.log,
                    {
                        "run": run_num,
                        "start_url": start_url,
                        "success": result.success,
                        "steps": result.steps,
                        "reason": result.reason,
                        "path": result.path,
                        "all_time_runs": run_count,
                        "all_time_successes": success_count,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )

                if args.runs == 0 or run_num < args.runs:
                    time.sleep(args.delay)

    except KeyboardInterrupt:
        print("\nStopped by user")

    print(f"\nDone. {successes}/{total} simulations succeeded this session.")
    if args.log:
        print(f"Log written to {args.log}")


if __name__ == "__main__":
    main()
