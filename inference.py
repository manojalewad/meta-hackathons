from fastapi import FastAPI
from models import Observation, Action, Reward
from environment import startupEnv
import uvicorn
app = FastAPI(title="Startup Business Simulator")
env = startupEnv()

@app.post("/reset", response_model=Observation)
def reset(payload: dict = {}):
    task = payload.get("task", "easy")
    obs = env.reset(task)
    return Observation(**obs)


@app.post("/step")
def step(action: Action):
    obs, reward = env.step(action.action)
    return {
        "observation": Observation(**obs),
        "reward": Reward(**reward),
    }


@app.get("/state")
def state():
    return env.state()


@app.get("/")
def root():
    return {
        "status": "running",
        "message": "Startup Business Simulator OpenEnv is live"
    }


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
