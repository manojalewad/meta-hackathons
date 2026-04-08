import json
import os
from typing import List, Optional

import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

BENCHMARK = "startup-business-simulator"
TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 30
SUCCESS_SCORE_THRESHOLD = 0.1


client = None
if API_KEY:
    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY,
        )
    except Exception as exc:
        print(f"[DEBUG] Failed to initialize OpenAI client: {exc}", flush=True)
        client = None


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def fallback_action(available_actions: List[str]) -> str:
    preferred = [
        "hireEngineer",
        "runAds",
        "improveProduct",
        "hireMarketing",
        "launchFeature",
    ]

    for action in preferred:
        if action in available_actions:
            return action

    return available_actions[0] if available_actions else "noWork"


def choose_action(observation: dict) -> str:
    available_actions = observation.get("available_actions", [])

    if not available_actions:
        return "noWork"

    if client is None:
        return fallback_action(available_actions)

    prompt = f"""
You are managing a startup company.

Current observation:
{json.dumps(observation, indent=2)}

Choose exactly one action from this list:
{available_actions}

Return only the action name and nothing else.
"""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert startup business manager.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
            max_tokens=20,
        )

        text = (completion.choices[0].message.content or "").strip()

        for action in available_actions:
            if action in text:
                return action

    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)

    return fallback_action(available_actions)


for task in TASKS:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        response = requests.post(
            f"{ENV_URL}/reset",
            json={"task": task},
            timeout=10,
        )
        response.raise_for_status()
        observation = response.json()

        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action = choose_action(observation)

            try:
                result = requests.post(
                    f"{ENV_URL}/step",
                    json={"action": action},
                    timeout=10,
                )
                result.raise_for_status()
                data = result.json()

                observation = data.get("observation", {})
                reward_obj = data.get("reward", {})

                reward = float(reward_obj.get("reward", 0.0))
                done = bool(reward_obj.get("done", False))
                score = float(reward_obj.get("task_score", 0.0))
                score = max(0.0, min(1.0, score))

                rewards.append(reward)
                steps_taken = step

                log_step(
                    step=step,
                    action=action,
                    reward=reward,
                    done=done,
                    error=None,
                )

            except Exception as exc:
                rewards.append(0.0)
                steps_taken = step

                log_step(
                    step=step,
                    action=action,
                    reward=0.0,
                    done=True,
                    error=str(exc),
                )
                break

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        rewards.append(0.0)
        log_step(
            step=1,
            action="noWork",
            reward=0.0,
            done=True,
            error=str(exc),
        )

    log_end(
        success=success,
        steps=steps_taken,
        score=score,
        rewards=rewards,
    )
