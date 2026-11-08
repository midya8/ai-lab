"""Tag-aware routing: combine category, urgency and tags into a queue + SLA.

Rules are ordered and data-driven (first match wins for the queue; every
matching rule can escalate priority). Security always outranks everything.
"""
from .tags import TagClassifier
from .urgency import Triage

PRIORITIES = ["low", "normal", "high", "critical"]
SLA_HOURS = {"low": 72, "normal": 24, "high": 8, "critical": 1}

# (tag, queue, minimum priority)
DEFAULT_RULES = [
    ("security", "security-response", "critical"),
    ("enterprise", "enterprise-support", "high"),
    ("performance", "sre-oncall", "normal"),
    ("mobile", "mobile-team", "normal"),
]


def _bump(priority, steps=1):
    return PRIORITIES[min(len(PRIORITIES) - 1, PRIORITIES.index(priority) + steps)]


def _at_least(a, b):
    return a if PRIORITIES.index(a) >= PRIORITIES.index(b) else b


def route_with_tags(base, tags, rules=DEFAULT_RULES):
    """base: dict from Triage.route(); tags: list of tag names."""
    priority = "high" if base["urgent"] else "normal"
    queue = base["queue"]
    matched = []
    for tag, rule_queue, min_pri in rules:
        if tag in tags:
            if not matched:
                queue = rule_queue
            matched.append(tag)
            priority = _at_least(priority, min_pri)
    # urgent + any tag-based escalation rule stacks one level (capped)
    if base["urgent"] and matched:
        priority = _bump(priority) if priority != "critical" else priority
    return {**base, "tags": list(tags), "queue": queue, "priority": priority,
            "sla_hours": SLA_HOURS[priority], "matched_rules": matched,
            "needs_human": base["needs_human"] or priority in ("high", "critical")}


class TaggedTriage:
    """Triage (category+urgency) plus tag classifier behind one route()."""

    def __init__(self, rules=DEFAULT_RULES, min_confidence=0.5):
        self.rules = rules
        self.triage = Triage(min_confidence)
        self.tagger = TagClassifier()

    def fit(self, records, val_records=None):
        self.triage.fit(records)
        texts, tags = [r["text"] for r in records], [r["tags"] for r in records]
        self.tagger.fit(texts, tags)
        if val_records:
            self.tagger.tune_thresholds([r["text"] for r in val_records], [r["tags"] for r in val_records])
        return self

    def route(self, text):
        return route_with_tags(self.triage.route(text), self.tagger.predict_tags(text), self.rules)
