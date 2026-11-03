from dataclasses import dataclass

# USD per 1M tokens (input, output). List prices written from memory: VERIFY before quoting.
PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.0),
}
GROUP_FIELDS = ("tenant_id", "user_id", "endpoint", "model", "stage")


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    try:
        price_in, price_out = PRICING_PER_1M[model]
    except KeyError as exc:
        raise ValueError(f"No price configured for model '{model}'") from exc
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000


@dataclass(frozen=True)
class UsageRecord:
    trace_id: str
    tenant_id: str
    user_id: str
    endpoint: str
    stage: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


@dataclass
class Totals:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


class UsageLedger:
    """Every LLM call is recorded with who made it, so spend can be grouped by
    tenant, user, endpoint, model or pipeline stage."""

    def __init__(self) -> None:
        self._records: list[UsageRecord] = []
        self._spend: dict[str, float] = {}

    def add(self, record: UsageRecord) -> None:
        self._records.append(record)
        self._spend[record.tenant_id] = self._spend.get(record.tenant_id, 0.0) + record.cost_usd

    def tenant_spend(self, tenant_id: str) -> float:
        return self._spend.get(tenant_id, 0.0)

    def totals(self, by: str, tenant_id: str | None = None) -> dict[str, Totals]:
        if by not in GROUP_FIELDS:
            raise ValueError(f"group by one of {GROUP_FIELDS}")
        out: dict[str, Totals] = {}
        for rec in self._records:
            if tenant_id is not None and rec.tenant_id != tenant_id:
                continue
            bucket = out.setdefault(str(getattr(rec, by)), Totals())
            bucket.calls += 1
            bucket.prompt_tokens += rec.prompt_tokens
            bucket.completion_tokens += rec.completion_tokens
            bucket.cost_usd += rec.cost_usd
        return out

    def trace_total(self, trace_id: str) -> Totals:
        total = Totals()
        for rec in self._records:
            if rec.trace_id == trace_id:
                total.calls += 1
                total.prompt_tokens += rec.prompt_tokens
                total.completion_tokens += rec.completion_tokens
                total.cost_usd += rec.cost_usd
        return total


@dataclass(frozen=True)
class BudgetDecision:
    action: str  # "allow", "downgrade" or "block"
    model: str
    spent: float
    budget: float


class BudgetTracker:
    """Tenant budget: allow, then switch to the cheaper model at `downgrade_at`
    of the budget, then block (HTTP 429)."""

    def __init__(
        self,
        ledger: UsageLedger,
        *,
        default_budget: float = 5.0,
        budgets: dict[str, float] | None = None,
        downgrade_at: float = 0.8,
        primary_model: str = "gpt-4o",
        fallback_model: str = "gpt-4o-mini",
    ) -> None:
        self._ledger = ledger
        self._default = default_budget
        self._budgets = budgets or {}
        self._downgrade_at = downgrade_at
        self._primary = primary_model
        self._fallback = fallback_model

    def budget_for(self, tenant_id: str) -> float:
        return self._budgets.get(tenant_id, self._default)

    def decide(self, tenant_id: str) -> BudgetDecision:
        spent = self._ledger.tenant_spend(tenant_id)
        budget = self.budget_for(tenant_id)
        if spent >= budget:
            return BudgetDecision("block", self._fallback, spent, budget)
        if spent >= self._downgrade_at * budget:
            return BudgetDecision("downgrade", self._fallback, spent, budget)
        return BudgetDecision("allow", self._primary, spent, budget)
