import json
import os
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

client = None

if API_KEY:
    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY,
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize OpenAI client: {e}")
        client = None
else:
    print("[ERROR] No API key found. Falling back to rule-based actions.")

TASKS = ["easy", "medium", "hard"]


def choose_action(observation):
    available_actions = observation.get("available_actions", ["runAds"])

    # If no client available, use simple fallback strategy
    if client is None:
        if "hireEngineer" in available_actions:
            return "hireEngineer"
        if "runAds" in available_actions:
            return "runAds"
        return available_actions[0]

    try:
        prompt = f"""
You are managing a startup business.

Current observation:
{json.dumps(observation, indent=2)}

Choose exactly one action from this list:
{available_actions}

Return only the action name and nothing else.
"""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert startup business strategist."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
        )

        text = response.choices[0].message.content.strip()

        for action in available_actions:
            if action in text:
                return action

    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}")

    # Safe fallback if model output invalid
    if "runAds" in available_actions:
        return "runAds"

    return available_actions[0]


for task in TASKS:
    print(f"[START] task={task}")

    try:
        reset_response = requests.post(
            f"{ENV_URL}/reset",
            json={"task": task},
            timeout=10,
        )
        reset_response.raise_for_status()
        observation = reset_response.json()

    except Exception as e:
        print(f"[ERROR] reset failed for task={task}: {e}")
        print(f"[END] task={task} score=0.0")
        continue

    done = False
    step_num = 0
    final_score = 0.0

    while not done and step_num < 30:
        action = choose_action(observation)

        try:
            step_response = requests.post(
                f"{ENV_URL}/step",
                json={"action": action},
                timeout=10,
            )
            step_response.raise_for_status()
            result = step_response.json()

            observation = result.get("observation", {})
            reward = result.get("reward", {})

            reward_value = reward.get("reward", 0.0)
            progress = reward.get("progress", 0.0)
            done = reward.get("done", False)
            final_score = reward.get("task_score", progress)

            print(
                f"[STEP] task={task} step={step_num} "
                f"action={action} reward={reward_value} "
                f"progress={progress} done={done}"
            )

        except Exception as e:
            print(f"[ERROR] step failed for task={task} step={step_num}: {e}")
            break

        step_num += 1

    print(f"[END] task={task} score={final_score}")
