import asyncio
import json
import os
from typing import List, Optional

import requests
from openai import OpenAI

# REQUIRED: use the evaluator-injected proxy variables exactly
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4.1-mini")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

BENCHMARK = "startup-business-simulator"
TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 10


MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "meta-llama/Llama-3.1-8B-Instruct"
)

client = None

if API_KEY:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )

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
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} "
        f"done={str(done).lower()} "
        f"error={error if error else 'null'}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    rewards: List[float],
) -> None:
    reward_text = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} "
        f"steps={steps} rewards={reward_text}",
        flush=True,
    )


def choose_action(
    task: str,
    observation: dict,
    previous_actions: List[str],
) -> str:
    available_actions = observation.get("available_actions", [])

    if not available_actions:
        return "run_ads"

    prompt = f"""
You are controlling a startup company.

Task:
{task}

Observation:
{json.dumps(observation)}

Recent actions:
{previous_actions[-3:]}

Choose exactly one action from:
{available_actions}

Return only the action name.
""".strip()

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert startup management agent."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=10,
        )

        text = (
            response.choices[0].message.content or ""
        ).strip()

        for action in available_actions:
            if action in text:
                return action

    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)

    return available_actions[0]


async def run_task(task: str) -> None:
    rewards: List[float] = []
    previous_actions: List[str] = []

    steps_taken = 0
    success = False
    done = False

    log_start(task)

    try:
        reset_response = requests.post(
            f"{ENV_URL}/reset",
            json={"task": task},
            timeout=30,
        )
        reset_response.raise_for_status()

        observation = reset_response.json()

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action = choose_action(
                task,
                observation,
                previous_actions,
            )
            previous_actions.append(action)

            reward = 0.0
            error = None

            try:
                step_response = requests.post(
                    f"{ENV_URL}/step",
                    json={"action": action},
                    timeout=30,
                )
                step_response.raise_for_status()

                payload = step_response.json()

                observation = payload["observation"]
                reward_info = payload["reward"]

                reward = float(
                    reward_info.get("reward", 0.0)
                )
                done = bool(
                    reward_info.get("done", False)
                )

            except Exception as exc:
                done = True
                error = str(exc)

            rewards.append(reward)
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
        success = len(rewards) > 0 and max(rewards) > 0.0

        log_end(
            success=success,
            steps=steps_taken,
            rewards=rewards,
        )


async def main() -> None:
    for task in TASKS:
        await run_task(task)


if __name__ == "__main__":
    asyncio.run(main())
