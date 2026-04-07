---
title: Startup Business Simulator
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---
# Startup Business Simulator (OpenEnv)

A real-world OpenEnv environment where an AI agent manages a startup company.

The agent must make business decisions such as hiring employees, improving the product, running advertisements, raising prices, lowering prices, and pitching investors.

The environment is designed to simulate realistic startup growth challenges including cash management, employee morale, customer growth, product quality, and marketing strategy.

## Features

* Real-world startup management simulation
* OpenEnv compliant API
* Supports reset(), step(), and state()
* Typed observation, action, and reward models
* Deterministic graders for multiple tasks
* Baseline inference script using OpenAI client
* Docker support
* Hugging Face Space deployment ready

## Tasks

### Easy

Reach 500 customers before month 12.

### Medium

Survive 18 months with employee morale greater than 50.

### Hard

Reach 2000 customers and maintain profit greater than 50000.

## Action Space

Available actions:

* hireEngineer
* hireMarketing
* runAds
* improveProduct
* launchFeature
* raisePrice
* lowerPrice
* pitchInvestor
* noWork

## Observation Space

Each observation contains:

* month
* cash
* customers
* employeeCount
* employeeMorale
* productQuality
* marketingLevel
* monthlyRevenue
* monthlyExpense
* currentEvent
* available_actions
* activeTask

## Reward System

The reward function provides feedback throughout the episode.

Positive rewards are given for:

* Increasing customers
* Improving morale
* Increasing product quality
* Growing revenue
* Maintaining positive cash flow
* Making progress toward task goals

Negative rewards are given for:

* Running out of cash
* Low morale
* Poor product quality
* Excessive spending
* Harmful pricing decisions
* Repeating ineffective actions

Task score is always normalized between 0.0 and 1.0.

## Setup

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Running Locally

Start the server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Test reset endpoint:

```bash
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d "{}"
```

Test state endpoint:

```bash
curl http://localhost:7860/state
```

## Docker

Build the Docker image:

```bash
docker build -t startup-env .
```

Run the container:

```bash
docker run -p 7860:7860 startup-env
```

## Validation

Run OpenEnv validation:

```bash
openenv validate
```

Run pre-submission validator:

```bash
./validate-submission.sh https://manojalewad-startup.hf.space .
```

## Inference

The project includes a baseline inference.py script.

Required environment variables:

```bash
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=openai/gpt-oss-120b
HF_TOKEN=your_token_here
```

Run inference:

```bash
python inference.py
```

## Example Baseline Scores

* Easy: 0.80
* Medium: 0.47
* Hard: 0.03

## Hugging Face Space

Deploy this project as a Docker-based Hugging Face Space.

Recommended Space tag:

```text
openenv
```

## Project Structure

```text
server/
  app.py
  __init__.py
Dockerfile
environment.py
grader.py
inference.py
models.py
openenv.yaml
pyproject.toml
README.md
requirements.txt
tasks.py
validate-submission.sh
```

## License

This project is intended for hackathon and educational use.
