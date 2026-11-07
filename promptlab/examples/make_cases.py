"""Regenerate examples/ticket_cases.jsonl (deterministic, synthetic tickets)."""
import json
import random
from pathlib import Path

OPENERS = ["Hi team,", "Hello,", "Urgent:", "Quick question -", "Not sure who to ask, but", "Following up from last week:"]
BODIES = {
    "billing": [
        "I was charged twice for my plan this month and need a refund.",
        "The invoice for March shows a payment I never authorised.",
        "My card was billed after I cancelled; please reverse it.",
        "Can you resend the invoice? The total looks higher than quoted.",
        "I need a refund for the duplicate charge on my statement.",
        "Why was my payment declined when the balance is fine?",
    ],
    "bug": [
        "The app crash happens every time I open the export screen.",
        "I get an error 500 when saving a report.",
        "The dashboard is broken after the latest update, nothing is not loading properly.",
        "An exception pops up on startup and then it freezes.",
        "Search is broken: it shows an error for any query with a hyphen.",
        "The mobile client freezes on the upload step.",
    ],
    "account": [
        "I forgot my password and the reset link never arrives.",
        "I am locked out after too many login attempts.",
        "How do I change the email address on my profile?",
        "Two-factor codes stopped working on my new phone, I cannot log in.",
        "Please help me recover my login, I lost access.",
        "I need to update the email address tied to my workspace.",
    ],
    "feature": [
        "We would love dark mode for the editor.",
        "Please add CSV export to the analytics page.",
        "Feature request: support for single sign-on with Okta.",
        "I wish the calendar view could show weekends optionally.",
        "Any plans for support for webhooks on new comments?",
        "It would be great if you could please add keyboard shortcuts.",
    ],
}

rng = random.Random(7)
rows = []
i = 0
for _ in range(15):
    for label in BODIES:
        i += 1
        text = f"{rng.choice(OPENERS)} {rng.choice(BODIES[label])}"
        rows.append({"id": f"t{i:03d}", "vars": {"ticket": text},
                     "checks": [{"type": "regex", "value": rf"\b{label}\b", "ignore_case": True}]})
out = Path(__file__).with_name("ticket_cases.jsonl")
out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
print(f"wrote {len(rows)} cases to {out}")
