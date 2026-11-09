"""Compare two audit JSON results (e.g. before/after a cleaning pass or a data refresh)."""

from __future__ import annotations


def diff_audits(old: dict, new: dict) -> dict:
    """Summarise what changed between two audits of the same dataset.

    Rows are matched by index, so this is meaningful when the row order is stable
    (a relabelling pass, a new run). If the row count changed, per-row sets are
    still reported but ``same_rows`` is False and they should be read with care.
    """
    o = {i["row"]: i for i in old["label_issues"]}
    n = {i["row"]: i for i in new["label_issues"]}
    resolved = sorted(set(o) - set(n))
    introduced = sorted(set(n) - set(o))
    persisting = sorted(set(o) & set(n))
    changed_suggestion = [r for r in persisting if o[r]["suggested"] != n[r]["suggested"]]
    old_rate = old["n_label_issues"] / max(old["n"], 1)
    new_rate = new["n_label_issues"] / max(new["n"], 1)
    return {
        "same_rows": old["n"] == new["n"],
        "rows": [old["n"], new["n"]],
        "flagged_rate": [old_rate, new_rate],
        "flagged_rate_delta": new_rate - old_rate,
        "cv_accuracy_delta": new["cv_accuracy"] - old["cv_accuracy"],
        "estimated_noise_rate_delta": new["estimated_noise_rate"] - old["estimated_noise_rate"],
        "resolved": resolved,
        "introduced": introduced,
        "persisting": persisting,
        "changed_suggestion": changed_suggestion,
        "conflicting_duplicate_groups_delta": len(new["conflicting_duplicate_groups"]) - len(old["conflicting_duplicate_groups"]),
        "exact_duplicate_groups_delta": len(new["exact_duplicate_groups"]) - len(old["exact_duplicate_groups"]),
        "outliers_delta": len(new["outliers"]) - len(old["outliers"]),
    }


def render_diff(d: dict) -> str:
    L = ["# Audit diff", ""]
    if not d["same_rows"]:
        L += [f"Warning: row count changed ({d['rows'][0]} -> {d['rows'][1]}); row indices may not line up.", ""]
    L.append(f"- flagged share: {d['flagged_rate'][0]:.1%} -> {d['flagged_rate'][1]:.1%} ({d['flagged_rate_delta']:+.1%})")
    L.append(f"- cross-validated accuracy: {d['cv_accuracy_delta']:+.3f}")
    L.append(f"- estimated noise rate: {d['estimated_noise_rate_delta']:+.1%}")
    L.append(f"- label issues resolved: {len(d['resolved'])}, new: {len(d['introduced'])}, still flagged: {len(d['persisting'])}")
    L.append(f"- conflicting duplicate groups: {d['conflicting_duplicate_groups_delta']:+d}; exact duplicate groups: {d['exact_duplicate_groups_delta']:+d}; outliers: {d['outliers_delta']:+d}")
    return "\n".join(L) + "\n"
