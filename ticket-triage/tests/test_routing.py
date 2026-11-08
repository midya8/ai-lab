import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from triage.routing import route_with_tags, TaggedTriage, SLA_HOURS
from triage.tags import generate_tagged


def base(urgent=False, queue="bug", human=False):
    return {"label": "bug", "urgent": urgent, "queue": queue, "needs_human": human}


def test_no_tags_keeps_base_queue_normal():
    r = route_with_tags(base(), [])
    assert r["queue"] == "bug" and r["priority"] == "normal" and r["sla_hours"] == 24
    assert not r["needs_human"]


def test_urgent_without_tags_is_high():
    assert route_with_tags(base(True, "bug:urgent"), [])["priority"] == "high"


def test_security_overrides_queue_and_is_critical():
    r = route_with_tags(base(), ["mobile", "security"])
    assert r["queue"] == "security-response" and r["priority"] == "critical"
    assert r["sla_hours"] == SLA_HOURS["critical"] and r["needs_human"]
    assert r["matched_rules"] == ["mobile", "security"] or set(r["matched_rules"]) == {"mobile", "security"}


def test_enterprise_urgent_escalates_to_critical():
    r = route_with_tags(base(True, "bug:urgent"), ["enterprise"])
    assert r["queue"] == "enterprise-support" and r["priority"] == "critical"


def test_enterprise_not_urgent_is_high():
    assert route_with_tags(base(), ["enterprise"])["priority"] == "high"


def test_custom_rules_and_first_match_queue():
    rules = [("mobile", "m", "low"), ("performance", "p", "normal")]
    r = route_with_tags(base(), ["performance", "mobile"], rules)
    assert r["queue"] == "m" and r["priority"] == "normal"


def test_end_to_end_security_ticket():
    d = generate_tagged(900, seed=3)
    m = TaggedTriage().fit(d[:600], d[600:750])
    r = m.route("I was charged twice for my plan. I suspect a data breach")
    assert "security" in r["tags"] and r["queue"] == "security-response"
    assert r["priority"] == "critical"
    plain = m.route("How do I export my data to CSV?")
    assert plain["priority"] in ("normal", "high") and "security" not in plain["tags"]
