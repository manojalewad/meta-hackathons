import json
import os
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

TASKS = ["easy", "medium", "hard"]


def choose_action(observation):
    try:
        prompt = f"""
You are managing a startup.
Current observation:
{json.dumps(observation, indent=2)}

Return exactly one action from available_actions.
"""
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        text = response.choices[0].message.content.strip()

        for action in observation.get("available_actions", []):
            if action in text:
                return action

    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}")

    return observation.get("available_actions", ["runAds"])[0]

for task in TASKS:
    print(f"[START] task={task}")

    try:
        observation = requests.post(
            f"{ENV_URL}/reset",
            json={"task": task},
            timeout=10
        ).json()
    except Exception as e:
        print(f"[ERROR] reset failed: {e}")
        continue

    done = False
    step_num = 0
    final_score = 0.0

    while not done and step_num < 30:
        action = choose_action(observation)

        try:
            response = requests.post(
                f"{ENV_URL}/step",
                json={"action": action},
                timeout=10
            ).json()

            observation = response["observation"]
            reward = response["reward"]

            done = reward["done"]
            final_score = reward["task_score"]

            print(
                f"[STEP] task={task} step={step_num} "
                f"action={action} reward={reward['reward']} "
                f"progress={reward['progress']} done={done}"
            )

        except Exception as e:
            print(f"[ERROR] step failed: {e}")
            break

        step_num += 1

    print(f"[END] task={task} score={final_score}")
