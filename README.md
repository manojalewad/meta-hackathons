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

## Tasks

### Easy
Reach 500 customers before month 12.

### Medium
Survive 18 months with morale > 50.

### Hard
Reach 2000 customers and profit > 50000.

## Action Space
- hireEngineer
- hireMarketing
- runAds
- improveProduct
- launchFeature
- raisePrice
- lowerPrice
- pitchInvestor
- noWork

## Observation Space
Contains:
- month
- cash
- customers
- morale
- revenue
- expense
- current event

## Setup

```bash
pip install -r requirements.txt
uvicorn app:app --reload
````

## Docker

```bash
docker build -t startup-env .
docker run -p 7860:7860 startup-env
```

## Validation

```bash
openenv validate
```

## Hugging Face Space

Tag the Space with `openenv`.