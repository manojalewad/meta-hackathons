

import json
import os
from typing import List, Optional

import requests
from openai import OpenAI

# ------------------------------------------------------------
# Environment variables
# ------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "meta-llama/Meta-Llama-3-8B-Instruct"
)
OPENENV_URL = os.getenv("OPENENV_URL", "http://localhost:7860")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

# ------------------------------------------------------------
# OpenAI client
# ------------------------------------------------------------

client = None

try:
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=API_BASE_URL.rstrip("/"),
    )
except Exception as exc:
    print(f"[DEBUG] Failed to initialize OpenAI client: {exc}", flush=True)

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

BENCHMARK = "startup-business-simulator"
TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 10
SUCCESS_THRESHOLD = 0.5

# ------------------------------------------------------------
# Logging helpers
# ------------------------------------------------------------

def log_start(task: str) -> None:
    print(
        f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}",
        flush=True,
    )


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_text = error if error else "null"

    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} "
        f"done={str(done).lower()} "
        f"error={error_text}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    rewards: List[float],
) -> None:
    rewards_text = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} "
        f"steps={steps} rewards={rewards_text}",
        flush=True,
    )

# ------------------------------------------------------------
# LLM Action Selection
# ------------------------------------------------------------

def choose_action(
    task: str,
    observation: dict,
    history: List[str],
) -> str:
    available_actions = observation.get("available_actions", [])

    if not available_actions:
        return "run_ads"

    # If OpenAI client failed, still continue safely
    if client is None:
        return available_actions[0]

    prompt = f"""
You are managing a startup business.

Task:
{task}

Current startup state:
{json.dumps(observation, indent=2)}

Recent actions:
{history[-3:]}

Available actions:
{available_actions}

Choose exactly one action from the available actions.
Return ONLY the action name.
""".strip()

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert startup business strategist. "
                        "Reply with exactly one valid action."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
            max_tokens=16,
        )

        text = (response.choices[0].message.content or "").strip()

        for action in available_actions:
            if action.lower() in text.lower():
                return action

    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)

    # Fallback if model output is invalid
    if "run_ads" in available_actions:
        return "run_ads"

    return available_actions[0]

# ------------------------------------------------------------
# Run one task
# ------------------------------------------------------------

def run_task(task: str) -> None:
    rewards: List[float] = []
    history: List[str] = []

    steps_taken = 0
    success = False

    log_start(task)

    try:
        reset_response = requests.post(
            f"{OPENENV_URL}/reset",
            json={"task": task},
            timeout=30,
        )
        reset_response.raise_for_status()

        observation = reset_response.json()
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action = choose_action(
                task=task,
                observation=observation,
                history=history,
            )

            reward = 0.0
            error = None

            try:
                step_response = requests.post(
                    f"{OPENENV_URL}/step",
                    json={"action": action},
                    timeout=30,
                )
                step_response.raise_for_status()

                payload = step_response.json()

                observation = payload["observation"]

                reward_obj = payload["reward"]
                reward = float(reward_obj.get("reward", 0.0))
                done = bool(reward_obj.get("done", False))

            except Exception as exc:
                done = True
                error = str(exc)

            rewards.append(reward)
            history.append(action)
            steps_taken = step

            log_step(
                step=step,
                action=action,
                reward=reward,
                done=done,
                error=error,
            )

    except Exception as exc:
        log_step(
            step=1,
            action="none",
            reward=0.00,
            done=True,
            error=str(exc),
        )

    finally:
        avg_reward = (
            sum(rewards) / len(rewards)
            if rewards
            else 0.0
        )

        success = avg_reward >= SUCCESS_THRESHOLD

        log_end(
            success=success,
            steps=steps_taken,
            rewards=rewards,
        )

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    for task in TASKS:
        run_task(task)


if __name__ == "__main__":
    main()
