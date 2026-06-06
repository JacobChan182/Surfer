"""Main agent loop: observe → reward → learn → act."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from surfer.bandit import BanditPolicy
from surfer.llm_critic import LLMCritic, SelectionMode, blend_rewards
from surfer.keywords import link_features
from surfer.reward import score_page
from surfer.weights_store import DEFAULT_WEIGHTS_PATH, WeightsStore, load_weights, save_weights
from surfer.wiki import PageNotFoundError, WikiClient, WikiLink, WikiPage, WikiRateLimitError, url_to_title


@dataclass
class StepRecord:
    step: int
    title: str
    reward: float
    matched_keywords: list[str]
    chosen_link: str | None = None
    llm_progress_score: float | None = None
    llm_reasoning: str | None = None


@dataclass
class RunResult:
    success: bool
    steps: int
    path: list[str]
    reason: str
    history: list[StepRecord] = field(default_factory=list)
    final_weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "steps": self.steps,
            "path": self.path,
            "reason": self.reason,
            "history": [
                {
                    "step": r.step,
                    "title": r.title,
                    "reward": r.reward,
                    "matched_keywords": r.matched_keywords,
                    "chosen_link": r.chosen_link,
                    "llm_progress_score": r.llm_progress_score,
                    "llm_reasoning": r.llm_reasoning,
                }
                for r in self.history
            ],
            "final_weights": self.final_weights,
        }


@dataclass
class AgentConfig:
    max_steps: int = 200
    verbose: bool = False
    use_llm: bool = False
    pick_mode: SelectionMode = "bandit"
    llm_candidate_limit: int = 0
    llm_blend: float = 0.6
    llm_delta_scale: float = 0.5
    weights_file: Path = DEFAULT_WEIGHTS_PATH
    save_weights: bool = True
    initial_weights: dict[str, float] | None = None
    weights_store: WeightsStore | None = None


def _select_next_link(
    *,
    page: WikiPage,
    path: list[str],
    last_link: str | None,
    keyword_reward: float,
    matched_keywords: list[str],
    candidates: list[WikiLink],
    bandit: BanditPolicy,
    critic: LLMCritic | None,
    pick_mode: SelectionMode,
    verbose: bool,
) -> tuple[WikiLink, frozenset[str], float, str | None, float | None]:
    """Select the next link and compute effective reward for the previous link update."""
    effective_reward = keyword_reward
    llm_reasoning: str | None = None
    llm_progress: float | None = None

    if pick_mode == "llm" and critic is not None:
        link_pool = bandit.rank_candidates(candidates, critic.candidate_limit)
        if verbose:
            if critic.candidate_limit <= 0:
                print(f"  -> llm link list: {len(link_pool)} links")
            elif len(candidates) > len(link_pool):
                print(f"  -> llm shortlist: {len(link_pool)} of {len(candidates)} links")

        chosen, pick_result = critic.pick_link(
            page,
            path,
            last_link,
            keyword_reward,
            matched_keywords,
            bandit.top_weights(10),
            link_pool,
        )
        if pick_result is not None:
            effective_reward = blend_rewards(
                keyword_reward,
                pick_result.progress_score,
                critic.llm_blend,
            )
            bandit.apply_deltas(pick_result.weight_updates, scale=critic.delta_scale)
            llm_reasoning = pick_result.reasoning
            llm_progress = pick_result.progress_score

        if chosen is None:
            if verbose:
                print("  [llm] pick failed, falling back to bandit")
            chosen, features = bandit.select(candidates)
        else:
            features = link_features(chosen.title)
    else:
        if critic is not None:
            critic_result = critic.evaluate(
                page,
                path,
                last_link,
                keyword_reward,
                matched_keywords,
                bandit.top_weights(10),
            )
            if critic_result is not None:
                effective_reward = blend_rewards(
                    keyword_reward,
                    critic_result.progress_score,
                    critic.llm_blend,
                )
                bandit.apply_deltas(critic_result.weight_updates, scale=critic.delta_scale)
                llm_reasoning = critic_result.reasoning
                llm_progress = critic_result.progress_score

        chosen, features = bandit.select(candidates)

    return chosen, features, effective_reward, llm_reasoning, llm_progress


def _fetch_page(
    client: WikiClient,
    current_title: str,
    path: list[str],
    visited: set[str],
    bandit: BanditPolicy,
    verbose: bool,
    max_repicks: int = 20,
) -> tuple[WikiPage, str, frozenset[str]]:
    """Fetch the current page, re-picking from the parent if the target is missing."""
    features: frozenset[str] = frozenset()
    repicks = 0

    while True:
        try:
            page = client.fetch(current_title)
            return page, current_title, features
        except PageNotFoundError:
            repicks += 1
            if verbose:
                print(f"  -> skipped missing page: {current_title}")
            visited.add(current_title)

            if len(path) <= 1 or repicks > max_repicks:
                raise

            path.pop()
            current_title = path[-1]
            parent = client.fetch(current_title)
            candidates = client.filter_unvisited(parent.links, visited)
            if not candidates:
                raise PageNotFoundError(f"No valid links from {current_title}")

            chosen, features = bandit.select(candidates)
            if verbose:
                print(f"  -> re-picked: {chosen.title}")
            visited.add(chosen.title)
            path.append(chosen.title)
            current_title = chosen.title


def run_agent(
    start_url: str,
    max_steps: int = 200,
    verbose: bool = False,
    wiki: WikiClient | None = None,
    critic: LLMCritic | None = None,
    config: AgentConfig | None = None,
) -> RunResult:
    """Run the closed-loop Wikipedia navigation agent."""
    cfg = config or AgentConfig(
        max_steps=max_steps,
        verbose=verbose,
        use_llm=critic is not None,
    )
    if config is None:
        cfg.max_steps = max_steps
        cfg.verbose = verbose

    owns_wiki = wiki is None
    client = wiki or WikiClient()

    stored = cfg.weights_store
    initial_weights = cfg.initial_weights or (stored.weights if stored else {})
    bandit = BanditPolicy(weights=dict(initial_weights))

    run_success = False
    run_steps = 0

    try:
        start_title = url_to_title(start_url)
        try:
            start_page = client.fetch(start_title)
        except PageNotFoundError:
            return RunResult(
                success=False,
                steps=0,
                path=[],
                reason=f"Start page not found: {start_title}",
            )
        except WikiRateLimitError as exc:
            return RunResult(
                success=False,
                steps=0,
                path=[],
                reason=str(exc),
            )

        if cfg.verbose and stored and stored.weights:
            top = sorted(stored.weights.items(), key=lambda kv: kv[1], reverse=True)[:5]
            loaded = ", ".join(f"{w}={v:.2f}" for w, v in top)
            print(f"Loaded distilled weights: {loaded}")

        visited: set[str] = {start_page.title}
        current_title = start_page.title
        path: list[str] = [start_page.title]
        history: list[StepRecord] = []
        last_features: frozenset[str] = frozenset()
        last_link: str | None = None
        page = start_page

        for step in range(1, cfg.max_steps + 1):
            if step > 1:
                try:
                    page, current_title, repick_features = _fetch_page(
                        client,
                        current_title,
                        path,
                        visited,
                        bandit,
                        cfg.verbose,
                    )
                    if repick_features:
                        last_features = repick_features
                except PageNotFoundError:
                    run_steps = step - 1
                    return RunResult(
                        success=False,
                        steps=step - 1,
                        path=path,
                        reason=f"Page not found: {current_title}",
                        history=history,
                        final_weights=dict(bandit.weights),
                    )
                except WikiRateLimitError as exc:
                    return RunResult(
                        success=False,
                        steps=step - 1,
                        path=path,
                        reason=str(exc),
                        history=history,
                        final_weights=dict(bandit.weights),
                    )

            result = score_page(page.title, page.extract)
            keyword_reward = result.reward

            record = StepRecord(
                step=step,
                title=page.title,
                reward=keyword_reward,
                matched_keywords=result.matched_keywords,
            )
            history.append(record)

            if cfg.verbose:
                kw = ", ".join(result.matched_keywords) or "none"
                print(f"Step {step} | {page.title} | reward={keyword_reward:.2f} | hits: {kw}")

            if result.is_terminal:
                run_success = True
                run_steps = step
                return RunResult(
                    success=True,
                    steps=step,
                    path=path,
                    reason="Reached Counter-Strike",
                    history=history,
                    final_weights=dict(bandit.weights),
                )

            if step == cfg.max_steps:
                run_steps = step
                return RunResult(
                    success=False,
                    steps=step,
                    path=path,
                    reason="Max steps reached",
                    history=history,
                    final_weights=dict(bandit.weights),
                )

            candidates = client.filter_unvisited(page.links, visited)
            if not candidates:
                run_steps = step
                return RunResult(
                    success=False,
                    steps=step,
                    path=path,
                    reason="Dead end — no unvisited links",
                    history=history,
                    final_weights=dict(bandit.weights),
                )

            chosen, features, effective_reward, llm_reasoning, llm_progress = _select_next_link(
                page=page,
                path=path,
                last_link=last_link,
                keyword_reward=keyword_reward,
                matched_keywords=result.matched_keywords,
                candidates=candidates,
                bandit=bandit,
                critic=critic,
                pick_mode=cfg.pick_mode,
                verbose=cfg.verbose,
            )

            if llm_progress is not None:
                record.llm_progress_score = llm_progress
                record.llm_reasoning = llm_reasoning

            if last_features:
                bandit.update(last_features, effective_reward)

            record.chosen_link = chosen.title

            if cfg.verbose:
                top = bandit.top_weights(3)
                top_str = ", ".join(f"{w}={v:.2f}" for w, v in top) or "none"
                picker = "llm" if cfg.pick_mode == "llm" else "bandit"
                print(f"  -> picked ({picker}): {chosen.title} | top weights: {top_str}")

            visited.add(chosen.title)
            path.append(chosen.title)
            current_title = chosen.title
            last_link = chosen.title
            last_features = features

        run_steps = cfg.max_steps
        return RunResult(
            success=False,
            steps=cfg.max_steps,
            path=path,
            reason="Max steps reached",
            history=history,
            final_weights=dict(bandit.weights),
        )
    finally:
        if cfg.save_weights:
            save_weights(
                bandit.weights,
                path=cfg.weights_file,
                success=run_success,
                steps=run_steps,
                existing=stored,
            )
        if owns_wiki:
            client.close()
