from typing import Dict

def easyGrading(state: Dict) -> float:
    customer_score = min(state["customers"] / 500.0, 1.0)
    cash_score = 1.0 if state["cash"] > 0 else 0.0
    return round((0.8 * customer_score) + (0.2 * cash_score), 4)
def mediumGrading(state: Dict) -> float:
    month_score = min(state["month"] / 18.0, 1.0)

    morale_score = min(max(state["employeeMorale"], 0) / 100.0, 1.0)
    cash_score = 1.0 if state["cash"] > 0 else 0.0
    return round((0.5 * month_score) + (0.3 * morale_score) + (0.2 * cash_score), 4)
def hardGrading(state: Dict) -> float:
    customers_score = min(state["customers"] / 2000.0, 1.0)
    profit = state["monthlyRevenue"] - state["monthlyExpense"]

    profit_score = min(max(profit, 0) / 50000.0, 1.0)
    return round((0.6 * customers_score) + (0.4 * profit_score), 4)


GRADERS = {
    "easy": easyGrading,

    "medium": mediumGrading,
    "hard": hardGrading,
}



