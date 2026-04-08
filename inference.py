#!/usr/bin/env python3
"""
inference.py — Startup Business Simulator OpenEnv Agent
=======================================================
Runs an LLM agent through all startup tasks and emits structured stdout logs.

Required environment variables:
    API_BASE_URL      LLM API endpoint
    MODEL_NAME        Model identifier
    HF_TOKEN          HuggingFace / API key
    API_KEY           Injected evaluator key
    ENV_BASE_URL      Environment server URL

Stdout format:
    [START] task=<task> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import os
import json
from typing import List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://router.huggingface.co/v1"
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "meta-llama/Llama-3.2-3B-Instruct"
)

ENV_BASE_URL = os.getenv(
    "ENV_BASE_URL",
    "http://localhost:8000"
)

BENCHMARK = "startup-business-simulator"

TASKS = [
    "easy",
    "medium",
    "hard",
]

MAX_STEPS = 10
SUCCESS_SCORE_THRESHOLD = 0.5
TEMPERATURE = 0.0
MAX_TOKENS = 32

SYSTEM_PROMPT = """
You are an expert startup business manager.

You are given the current state of a startup company.

You must choose exactly one action from the available actions.

Reply with ONLY the action name.
Do not explain your answer.
Do not output extra text.
""".strip()

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(
        f"[START] task={task} env={env} model={model}",
        flush=True,
    )


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"

    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} "
        f"steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def get_model_action(
    client: OpenAI,
    task: str,
    observation: dict,
    history: List[str],
) -> str:
    available_actions = observation.get("available_actions", [])

    if not available_actions:
        return "run_ads"

    history_block = "\n".join(history[-3:]) if history else "None"

    user_prompt = f"""
Task:
{task}

Current startup state:
{json.dumps(observation, indent=2)}

Recent decisions:
{history_block}

Available actions:
{available_actions}

Return exactly one action from the list above.
""".strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )

        text = (
            completion.choices[0].message.content or ""
        ).strip()

        for action in available_actions:
            if action.lower() in text.lower():
                return action

        return available_actions[0]

    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return available_actions[0]

# ---------------------------------------------------------------------------
# Single task runner
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task_name: str) -> None:
    history: List[str] = []
    rewards: List[float] = []

    steps_taken = 0
    score = 0.0
    success = False
    last_error: Optional[str] = None

    log_start(
        task=task_name,
        env=BENCHMARK,
        model=MODEL_NAME,
    )

    try:
        response = requests.post(
            f"{ENV_BASE_URL}/reset",
            json={"task": task_name},
            timeout=30,
        )
        response.raise_for_status()

        observation = response.json()
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action = get_model_action(
                client=client,
                task=task_name,
                observation=observation,
                history=history,
            )

            try:
                step_response = requests.post(
                    f"{ENV_BASE_URL}/step",
                    json={"action": action},
                    timeout=30,
                )
                step_response.raise_for_status()

                result = step_response.json()

                observation = result["observation"]

                reward_info = result["reward"]
                reward = float(reward_info.get("reward", 0.0))
                done = bool(reward_info.get("done", False))

                last_error = None

            except Exception as exc:
                reward = 0.0
                done = True
                last_error = str(exc)

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action,
                reward=reward,
                done=done,
                error=last_error,
            )

            history.append(
                f"Step {step}: {action} -> reward {reward:.2f}"
            )

            if done:
                break

        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = max(0.0, min(score, 1.0))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        last_error = str(exc)
        print(f"[DEBUG] Task {task_name} failed: {last_error}", flush=True)

    finally:
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    for task_name in TASKS:
        run_task(client, task_name)


if __name__ == "__main__":
    main()
