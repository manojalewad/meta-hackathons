
import os
import json
import asyncio
from typing import List, Optional

import requests
from openai import OpenAI


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = os.getenv("API_KEY")

TOKEN = API_KEY or HF_TOKEN
if TOKEN is None:
    raise ValueError("Missing API_KEY/HF_TOKEN environment variable")

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

BENCHMARK = "startup-business-simulator"
TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 10
SUCCESS_SCORE_THRESHOLD = 0.5

try:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=TOKEN,
    )
except Exception as exc:
    print(f"[ERROR] failed_to_initialize_client={exc}", flush=True)
    raise


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)



def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_value = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_value}",
        flush=True,
    )



def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    reward_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={reward_str}",
        flush=True,
    )



def build_prompt(task: str, observation: dict, previous_actions: List[str]) -> str:
    recent_actions = previous_actions[-3:] if previous_actions else []

    return f"""
You are managing a startup company.

Current task: {task}

Current company state:
{json.dumps(observation, indent=2)}

Recent actions:
{recent_actions}

Choose exactly one action from available_actions.

Strategy:
- easy: focus on run_ads until customer target is reached
- medium: use improve_product and hire_engineer, then run_ads
- hard: combine improve_product, hire_engineer, hire_marketing, and run_ads
- avoid repeating the same action more than 2 times in a row
- never invent new actions

Return only the action name.
""".strip()



def choose_action(task: str, observation: dict, previous_actions: List[str]) -> str:
    prompt = build_prompt(task, observation, previous_actions)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI agent that chooses business actions for a startup simulation.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=20,
        )

        text = (response.choices[0].message.content or "").strip()

        for action in observation.get("available_actions", []):
            if action in text:
                return action

    except Exception:
        pass

    available = observation.get("available_actions", [])

    if task == "easy" and "run_ads" in available:
        return "run_ads"
    if task == "medium" and "improve_product" in available:
        return "improve_product"
    if task == "hard" and "hire_engineer" in available:
        return "hire_engineer"

    return available[0] if available else "run_ads"


async def run_task(task: str) -> None:
    rewards: List[float] = []
    previous_actions: List[str] = []
    steps_taken = 0
    final_score = 0.0
    success = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        response = requests.post(
            f"{ENV_URL}/reset",
            json={"task": task},
            timeout=30,
        )
        response.raise_for_status()
        observation = response.json()

        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action = choose_action(task, observation, previous_actions)
            previous_actions.append(action)

            error = None

            try:
                result = requests.post(
                    f"{ENV_URL}/step",
                    json={"action": action},
                    timeout=30,
                )
                result.raise_for_status()
                payload = result.json()

                observation = payload["observation"]
                reward_info = payload["reward"]

                reward = float(reward_info.get("reward", 0.0))
                done = bool(reward_info.get("done", False))
                final_score = float(reward_info.get("score", 0.0))

            except Exception as exc:
                reward = 0.0
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

            if done:
                break

        success = final_score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(
            success=success,
            steps=steps_taken,
            score=final_score,
            rewards=rewards,
        )


async def main() -> None:
    for task in TASKS:
        await run_task(task)


if __name__ == "__main__":
    asyncio.run(main())

