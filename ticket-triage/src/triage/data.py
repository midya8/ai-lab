"""Synthetic support-ticket generator (deterministic)."""
import random

TEMPLATES = {
    "billing": ["I was charged twice for my {x}", "Please refund my last {x} payment", "My invoice shows the wrong amount for {x}", "Credit card declined when paying for {x}"],
    "bug": ["The app crashes when I open {x}", "Error 500 appears on the {x} page", "{x} is broken after the latest update", "Button on {x} does nothing and throws an exception"],
    "account": ["I cannot log in to my {x}", "Reset my password for {x} please", "How do I change the email on my {x}", "My {x} is locked after failed sign in attempts"],
    "feature": ["Would be great to have dark mode in {x}", "Please add export to CSV for {x}", "Feature request: support integration with {x}", "Can you add a shortcut for {x}"],
}
NOUNS = ["subscription", "dashboard", "profile", "report", "workspace", "mobile app", "settings", "checkout", "plan", "team"]
PRIORITY_HIGH = ["urgent", "asap", "production down", "immediately", "blocking our team"]


def generate(n=400, seed=0):
    rng = random.Random(seed)
    out = []
    labels = list(TEMPLATES)
    for _ in range(n):
        lab = rng.choice(labels)
        text = rng.choice(TEMPLATES[lab]).format(x=rng.choice(NOUNS))
        urgent = rng.random() < 0.25
        if urgent:
            text += " " + rng.choice(PRIORITY_HIGH)
        out.append({"text": text, "category": lab, "urgent": urgent})
    return out
