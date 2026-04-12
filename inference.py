import json
import os
from typing import List, Optional

import httpx
import requests
from openai import OpenAI

# ------------------------------------------------------------
# Environment variables — read ALL at module load time
# ------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
API_BASE_URL   = os.getenv("API_BASE_URL")        # REQUIRED — no default allowed
MODEL_NAME     = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
OPENENV_URL    = os.getenv("OPENENV_URL", "http://localhost:7860")

# ------------------------------------------------------------
# Hard validation — crash immediately with a clear message
# ------------------------------------------------------------

if not OPENAI_API_KEY:
    raise RuntimeError(
        "[FATAL] No API key found. "
        "Set the OPENAI_API_KEY (or API_KEY) environment variable."
    )

if not API_BASE_URL:
    raise RuntimeError(
        "[FATAL] API_BASE_URL is not set. "
        "You must inject it via environment variable — no hardcoded or default URL is allowed."
    )

# Sanitise: strip surrounding whitespace/quotes and trailing slash
API_BASE_URL = API_BASE_URL.strip().strip('"').strip("'").rstrip("/")

if not API_BASE_URL.startswith("http"):
    raise RuntimeError(
        f"[FATAL] API_BASE_URL looks invalid: '{API_BASE_URL}'. "
        "It must start with http:// or https://"
    )

print(f"[CONFIG] API_BASE_URL = {API_BASE_URL}", flush=True)
print(f"[CONFIG] MODEL_NAME   = {MODEL_NAME}",   flush=True)

# ------------------------------------------------------------
# OpenAI client — module-level, never inside a function or main()
# Explicit httpx client avoids SyncHttpxClientWrapper version issues
# ------------------------------------------------------------

_http_client = httpx.Client(
    timeout=httpx.Timeout(60.0, connect=10.0),
    follow_redirects=True,
)

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=API_BASE_URL,
    http_client=_http_client,
)

print("[CONFIG] OpenAI client initialised successfully.", flush=True)

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

BENCHMARK         = "startup-business-simulator"
TASKS             = ["easy", "medium", "hard"]
MAX_STEPS         = 10
SUCCESS_THRESHOLD = 0.5

# Score must be strictly inside (0, 1) — never exactly 0.0 or 1.0
SCORE_MIN = 1e-6
SCORE_MAX = 1.0 - 1e-6


def clamp_score(value: float) -> float:
    """Clamp a score to the open interval (0, 1) exclusively."""
    return max(SCORE_MIN, min(SCORE_MAX, value))

# ------------------------------------------------------------
# Logging helpers
# ------------------------------------------------------------

def log_start(task: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.6f} "
        f"done={str(done).lower()} "
        f"error={error if error else 'null'}",
        flush=True,
    )


def log_end(score: float, steps: int, rewards: List[float]) -> None:
    """Log final score — always strictly within (0, 1)."""
    print(
        f"[END] score={score:.6f} "
        f"steps={steps} "
        f"rewards={','.join(f'{r:.6f}' for r in rewards)}",
        flush=True,
    )

# ------------------------------------------------------------
# LLM action selection — MUST make a real API call every time.
# No try/except, no fallback, no client=None path.
# ------------------------------------------------------------

def choose_action(task: str, observation: dict, history: List[str]) -> str:
    available_actions = observation.get("available_actions", [])

    if not available_actions:
        raise ValueError("Observation contains no available_actions.")

    prompt = f"""You are managing a startup business.

Task:
{task}

Current startup state:
{json.dumps(observation, indent=2)}

Recent actions:
{history[-3:]}

Available actions:
{available_actions}

Choose exactly one action from the list above.
Return ONLY the action name — no explanation, no punctuation."""

    # Real API call — must route through API_BASE_URL.
    # Exceptions propagate; there is no silent fallback.
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert startup business strategist. "
                    "Reply with exactly one valid action name and nothing else."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=16,
    )

    text = (response.choices[0].message.content or "").strip()
    print(f"[LLM] response: {repr(text)}", flush=True)

    for action in available_actions:
        if action.lower() in text.lower():
            return action

    raise ValueError(
        f"Model returned '{text}' which does not match any available action: {available_actions}"
    )

# ------------------------------------------------------------
# Run one task — returns a score strictly in (0, 1)
# ------------------------------------------------------------

def run_task(task: str) -> float:
    rewards: List[float] = []
    history: List[str]   = []
    steps_taken          = 0

    log_start(task)

    try:
        reset_resp = requests.post(
            f"{OPENENV_URL}/reset",
            json={"task": task},
            timeout=30,
        )
        reset_resp.raise_for_status()

        observation = reset_resp.json()
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action = choose_action(task=task, observation=observation, history=history)

            reward = 0.0
            error  = None

            try:
                step_resp = requests.post(
                    f"{OPENENV_URL}/step",
                    json={"action": action},
                    timeout=30,
                )
                step_resp.raise_for_status()

                payload     = step_resp.json()
                observation = payload["observation"]
                reward_obj  = payload["reward"]
                reward      = float(reward_obj.get("reward", 0.0))
                done        = bool(reward_obj.get("done", False))

            except Exception as exc:
                done  = True
                error = str(exc)

            rewards.append(reward)
            history.append(action)
            steps_taken = step

            log_step(step=step, action=action, reward=reward, done=done, error=error)

    except Exception as exc:
        log_step(step=1, action="none", reward=0.0, done=True, error=str(exc))

    finally:
        raw_score = sum(rewards) / len(rewards) if rewards else 0.0

        # CRITICAL: clamp to strictly open interval (0, 1)
        # A raw score of exactly 0.0 or 1.0 will be rejected by the evaluator
        score = clamp_score(raw_score)

        log_end(score=score, steps=steps_taken, rewards=rewards)

    return score

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    for task in TASKS:
        score = run_task(task)
        print(f"[SCORE] task={task} score={score:.6f}", flush=True)
        assert SCORE_MIN <= score <= SCORE_MAX, (
            f"Score {score} is out of range — must be strictly between 0 and 1."
        )


if __name__ == "__main__":
    main()
