from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class Observation(BaseModel):
    month: int
    cash: float
    customers: int
    employeeCount: int
    employeeMorale: float
    productQuality: float
    marketingLevel: float
    monthlyRevenue: float
    monthlyExpense: float
    currentEvent: str
    available_actions: List[str]
    activeTask: str


class Action(BaseModel):
    action: str = Field(..., description="Action selected by the agent")

class Reward(BaseModel):
    reward: float = Field(..., ge=0.0, le=1.0)
    done: bool
    progress: float = Field(..., ge=0.0, le=1.0)
    task_score: float = Field(..., ge=0.0, le=1.0)
    info: Dict
