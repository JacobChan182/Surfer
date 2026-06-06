"""CLI entry point for the Wikipedia Surfer agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from surfer.env import load_env
from surfer.agent import AgentConfig, run_agent
from surfer.llm_critic import LLMCritic
from surfer.nvidia_llm import DEFAULT_MODEL, NvidiaLLM, NvidiaLLMError
from surfer.weights_store import DEFAULT_WEIGHTS_PATH, load_weights, print_top_weights, reset_weights


def main() -> None:
    load_env()

    parser = argparse.ArgumentParser(
        description="Navigate Wikipedia toward Counter-Strike using a self-optimizing bandit."
    )
    parser.add_argument(
        "--start",
        help='Starting Wikipedia URL (e.g. "https://en.wikipedia.org/wiki/Dog")',
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum page attempts before giving up (default: 200)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-step details including matched keywords and top weights",
    )
    parser.add_argument(
        "--json-out",
        metavar="PATH",
        help="Write full run trace to a JSON file",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable NVIDIA LLM critic for progress scoring (bandit picks links unless --pick llm)",
    )
    parser.add_argument(
        "--pick",
        choices=("bandit", "llm"),
        default="bandit",
        help="Link selection: bandit (default) or llm (LLM chooses from candidate list)",
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
        help=f"NVIDIA NIM model id (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--llm-blend",
        type=float,
        default=0.6,
        help="Blend factor for LLM progress score vs keyword reward (default: 0.6)",
    )
    parser.add_argument(
        "--llm-delta-scale",
        type=float,
        default=0.5,
        help="Scale factor for LLM weight_updates (default: 0.5)",
    )
    parser.add_argument(
        "--weights-file",
        type=Path,
        default=DEFAULT_WEIGHTS_PATH,
        help=f"Path to distilled weights JSON (default: {DEFAULT_WEIGHTS_PATH})",
    )
    parser.add_argument(
        "--no-save-weights",
        action="store_true",
        help="Do not persist weights after the run",
    )
    parser.add_argument(
        "--reset-weights",
        action="store_true",
        help="Delete saved weights before starting",
    )
    parser.add_argument(
        "--top-weights",
        action="store_true",
        help="Print the top 15 saved keyword weights and exit",
    )
    args = parser.parse_args()

    if args.top_weights:
        print_top_weights(args.weights_file, n=15)
        return

    if not args.start:
        parser.error("--start is required unless using --top-weights")

    if args.reset_weights:
        reset_weights(args.weights_file)
        if args.verbose:
            print(f"Reset weights at {args.weights_file}")

    stored = load_weights(args.weights_file)

    critic: LLMCritic | None = None
    if args.llm or args.pick == "llm":
        try:
            llm = NvidiaLLM(model=args.llm_model)
            critic = LLMCritic(
                llm=llm,
                llm_blend=args.llm_blend,
                delta_scale=args.llm_delta_scale,
                candidate_limit=args.llm_candidate_limit,
                verbose=args.verbose,
            )
        except NvidiaLLMError as exc:
            print(f"LLM setup failed: {exc}", file=sys.stderr)
            sys.exit(1)

    if args.pick == "llm" and critic is None:
        print("LLM pick mode requires --llm or a valid NVIDIA_API_KEY", file=sys.stderr)
        sys.exit(1)

    config = AgentConfig(
        max_steps=args.max_steps,
        verbose=args.verbose,
        use_llm=args.llm or args.pick == "llm",
        pick_mode=args.pick,
        llm_candidate_limit=args.llm_candidate_limit,
        llm_blend=args.llm_blend,
        llm_delta_scale=args.llm_delta_scale,
        weights_file=args.weights_file,
        save_weights=not args.no_save_weights,
        weights_store=stored,
    )

    result = run_agent(
        start_url=args.start,
        critic=critic,
        config=config,
    )

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

    if result.success:
        print(f"\nSuccess in {result.steps} steps!")
        print("Path:", " -> ".join(result.path))
    else:
        print(f"\nFailed after {result.steps} steps: {result.reason}")
        print("Path:", " -> ".join(result.path))
        sys.exit(1)


if __name__ == "__main__":
    main()
