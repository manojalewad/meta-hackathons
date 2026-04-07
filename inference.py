import json
import os
import requests
from openai import OpenAI

API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ["MODEL_NAME"]
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

TASKS = ["easy", "medium", "hard"]


def choose_action(observation):
    prompt = f"""
You are managing a startup business.

Current task: {observation["active_task"]}

Current state:
{json.dumps(observation, indent=2)}

Choose exactly ONE action from:
{", ".join(observation["available_actions"])}

Strategy:
- easy: prioritize run_ads
- medium: balance hire_engineer, improve_product, run_ads
- hard: improve_product first, then run_ads, then hire_engineer
- avoid repeating the same action many times

Return only the action name.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a business planning agent."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=20,
    )

    text = response.choices[0].message.content.strip()

    for action in observation["available_actions"]:
        if action in text:
            return action

    return observation["available_actions"][0]


for task in TASKS:
    reset_response = requests.post(
        f"{ENV_URL}/reset",
        json={"task": task},
    )
    observation = reset_response.json()

    print(
        f"[START] task={task} env=startup-business-simulator model={MODEL_NAME}"
    )

    rewards = []
    done = False
    step = 0
    final_score = 0.0

    while not done and step < 10:
        action = choose_action(observation)

        step_response = requests.post(
            f"{ENV_URL}/step",
            json={"action": action},
        )
        data = step_response.json()

        observation = data["observation"]
        reward = float(data["reward"]["reward"])
        done = bool(data["reward"]["done"])
        final_score = float(data["reward"]["score"])

        rewards.append(reward)
        step += 1

        print(
            f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null"
        )

    reward_text = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(final_score >= 0.5).lower()} "
        f"steps={step} score={final_score:.3f} rewards={reward_text}"
    )
