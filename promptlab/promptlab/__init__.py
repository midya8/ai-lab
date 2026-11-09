"""promptlab: prompt versioning, paired A/B comparison and regression gates."""
from .template import Template, TemplateError
from .registry import Registry, RegistryError, Version
from .checks import run_check, CheckResult
from .runner import Case, CaseResult, RunResult, load_cases, run_prompt
from .compare import compare_runs, check_regression, render_markdown, Comparison, RegressionReport
from .stats import wilson, mcnemar_exact, paired_bootstrap, cohen_kappa, spearman
from .models import RuleModel, CommandModel, RuleJudge, make_model, make_judge
from .judge import (build_judge_prompt, parse_judge_reply, judge_score, calibrate_judge,
                    load_calibration_items, render_calibration, JudgeError)

from .pairwise import (build_pairwise_prompt, parse_pairwise_reply, judge_pair, compare_outputs, summarize,
                       audit_judge, load_pair_items, make_pair_judge, PairVerdict, PairwiseError,
                       RulePairJudge, PositionBiasedJudge)

from .ensemble import compare_ensemble, render_ensemble, fleiss_kappa, EnsembleResult, ensemble_gate

__version__ = "0.5.0"
