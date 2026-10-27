"""A small forward-chaining, rule-based expert system (educational demo, not medical advice)."""
from __future__ import annotations

from dataclasses import dataclass, field

SYMPTOMS: dict[str, str] = {
    "fever": "Fever",
    "cough": "Cough",
    "cold": "Cold / runny nose",
    "sore_throat": "Sore throat",
    "headache": "Headache",
    "body_ache": "Body ache",
    "breathless": "Shortness of breath",
}


@dataclass(frozen=True)
class Rule:
    """IF all ``conditions`` are present THEN ``conclusion``."""

    id: str
    conclusion: str
    conditions: frozenset[str]
    advice: str = ""
    urgent: bool = False

    def describe(self) -> str:
        return " AND ".join(SYMPTOMS.get(c, c) for c in sorted(self.conditions))


# Order = priority (like if / elif). R2, R3 and R5 are the three rules from the original lab program.
DEFAULT_RULES: list[Rule] = [
    Rule("R1", "Seek urgent medical care", frozenset({"breathless"}),
         "Difficulty breathing should be checked by a doctor straight away.", urgent=True),
    Rule("R2", "Flu", frozenset({"fever", "cough"}), "Rest, fluids, and see a doctor if it worsens."),
    Rule("R3", "Common Cold", frozenset({"cold", "cough"}), "Rest and fluids usually help."),
    Rule("R4", "Viral fever", frozenset({"fever", "body_ache"}), "Rest, fluids and monitor your temperature."),
    Rule("R5", "Infection", frozenset({"fever"}), "A persistent fever deserves a medical check."),
    Rule("R6", "Sore throat / throat irritation", frozenset({"sore_throat"}), "Warm fluids may soothe it."),
    Rule("R7", "Headache", frozenset({"headache"}), "Rest and hydration; see a doctor if severe or frequent."),
]


@dataclass
class Diagnosis:
    primary: Rule | None
    fired: list[Rule] = field(default_factory=list)
    trace: list[tuple[Rule, bool]] = field(default_factory=list)


def diagnose(symptoms, rules: list[Rule] | None = None) -> Diagnosis:
    """Evaluate every rule; the first rule that fires (in priority order) is the primary result."""
    rules = DEFAULT_RULES if rules is None else rules
    present = set(symptoms)
    trace = [(r, bool(r.conditions) and r.conditions <= present) for r in rules]
    fired = [r for r, ok in trace if ok]
    return Diagnosis(fired[0] if fired else None, fired, trace)
