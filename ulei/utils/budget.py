"""
Budget control implementation for evaluation cost management.

Provides cost estimation, budget tracking, and enforcement of hard stops
when budget limits are exceeded during evaluation runs.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class BudgetManager:
    """
    Tracks and enforces budget limits for evaluation runs.
    """

    def __init__(self, budget_limit: Optional[float] = None):
        self.budget_limit = budget_limit
        self.current_cost = 0.0

    def add_cost(self, cost: float) -> None:
        """
        Add cost to the current total and check against budget.
        Raises BudgetExceededError if limit is exceeded.
        """
        self.current_cost += cost
        logger.debug(f"Added cost: {cost:.4f}, total: {self.current_cost:.4f}")
        if self.budget_limit is not None and self.current_cost > self.budget_limit:
            raise BudgetExceededError(
                f"Budget exceeded: {self.current_cost:.2f} > {self.budget_limit:.2f}",
                budget_limit=self.budget_limit,
                current_cost=self.current_cost,
            )

    def reset(self) -> None:
        """Reset the current cost to zero."""
        self.current_cost = 0.0

    def get_utilization(self) -> float:
        """
        Return the fraction of budget used (0.0-1.0), or 1.0 if no limit.
        """
        if self.budget_limit is None or self.budget_limit == 0:
            return 1.0
        return min(self.current_cost / self.budget_limit, 1.0)


class BudgetExceededError(Exception):
    """Exception raised when budget limit is exceeded."""

    budget_limit: Optional[float]
    current_cost: Optional[float]

    def __init__(
        self,
        message: str,
        budget_limit: Optional[float] = None,
        current_cost: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.budget_limit = budget_limit
        self.current_cost = current_cost
