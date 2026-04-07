import json
import os
import requests
from openai import OpenAI

API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.environ["MODEL_NAME"]
HF_TOKEN = os.environ["HF_TOKEN"]
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

TASKS = ["easy", "medium", "hard"]


def choose_action(observation):
    prompt = f"""
You are managing a startup.
Current observation:\n{json.dumps(observation, indent=2)}
Return exactly one action from available_actions.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    text = response.choices[0].message.content.strip()
    for action in observation["available_actions"]:
        if action in text:
            return action

    return "runAds"


for task in TASKS:
    print(f"[START] task={task}")

    observation = requests.post(
        f"{ENV_URL}/reset",
        json={"task": task}
    ).json()

    done = False
    step_num = 0
    final_score = 0.0

    while not done and step_num < 30:
        action = choose_action(observation)

        response = requests.post(
            f"{ENV_URL}/step",
            json={"action": action}
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

        step_num += 1

    print(f"[END] task={task} score={final_score}")