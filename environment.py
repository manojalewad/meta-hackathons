import random
from tasks import TASKS
from grader import GRADERS
class startupEnv:
    ACTIONS = [
        "hireEngineer",
        "hireMarketing",
        "runAds",
        "improveProduct",
        "launchFeature",
        "raisePrice",
        "lowerPrice",
        "pitchInvestor",
        "noWork",
    ]
    def __init__(self):
        self.task = "easy"
        self.reset(self.task)
    def reset(self, task="easy"):
        self.task = task
        self.max_months = TASKS[task]["max_months"]

        self.state_data = {
            "month": 1,
            "cash": 500000.0,
            "customers": 100,
            "employeeCount": 3,
            "employeeMorale": 75.0,
            "productQuality": 50.0,
            "marketingLevel": 25.0,
            "monthlyRevenue": 30000.0,
            "monthlyExpense": 25000.0,
            "currentEvent": f"Task started: {TASKS[task]['name']}"
        }
        return self._observation()
    def state(self):
        return self.state_data
    def _observation(self):
        return {
            **self.state_data,
            "available_actions": self.ACTIONS,
            "activeTask": self.task,
        }
    def step(self, action: str):
        s = self.state_data
        reward = 0.0
        if action == "hireEngineer":
            s["cash"] -= 20000
            s["employeeCount"] += 1
            s["productQuality"] += 8
            reward += 0.05
        elif action == "hireMarketing":
            s["cash"] -= 15000
            s["employeeCount"] += 1
            s["marketingLevel"] += 10
            reward += 0.04
        elif action == "runAds":
            s["cash"] -= 25000
            gained = random.randint(40, 120)
            s["customers"] += gained
            reward += min(gained / 500.0, 0.15)
        elif action == "improveProduct":
            s["cash"] -= 15000
            s["productQuality"] += 12
            reward += 0.05
        elif action == "launchFeature":
            s["cash"] -= 40000
            gained = random.randint(50, 180)
            s["customers"] += gained
            s["productQuality"] += 15
            reward += 0.10
        elif action == "raisePrice":
            s["monthlyRevenue"] += 10000
            s["customers"] -= random.randint(5, 20)
            reward += 0.03
        elif action == "lowerPrice":
            s["monthlyRevenue"] -= 5000
            s["customers"] += random.randint(20, 60)
            reward += 0.04
        elif action == "pitchInvestor":
            if random.random() < 0.5:
                funding = random.randint(100000, 300000)
                s["cash"] += funding
                s["currentEvent"] = f"Investor funded startup with ${funding}"
                reward += 0.15
            else:
                s["currentEvent"] = "Investor rejected the pitch"
                reward -= 0.05
        elif action == "noWork":
            reward -= 0.03
        profit = (
            s["customers"] * (1 + s["productQuality"] / 100.0)
            - s["employeeCount"] * 12000
            + s["marketingLevel"] * 100
        )
        s["monthlyRevenue"] = max(10000.0, profit + 30000)
        s["monthlyExpense"] = 15000 + s["employeeCount"] * 10000
        monthly_profit = s["monthlyRevenue"] - s["monthlyExpense"]
        s["cash"] += monthly_profit
        if random.random() < 0.1:
            lost = random.randint(20, 100)
            s["customers"] = max(0, s["customers"] - lost)
            s["currentEvent"] = f"Competitor stole {lost} customers"
            reward -= 0.04
        s["month"] += 1
        task_score = GRADERS[self.task](s)
        reward = max(0.0, min(1.0, reward + task_score * 0.5))
        done = False
        if s["cash"] <= 0:
            done = True
            reward = 0.0

        if s["month"] > self.max_months:
            done = True
        info = {
            "task": self.task,
            "task_goal": TASKS[self.task]["goal"],
        }
        return self._observation(), {
            "reward": round(reward, 4),
            "done": done,
            "progress": task_score,
            "task_score": task_score,
            "info": info,
        }