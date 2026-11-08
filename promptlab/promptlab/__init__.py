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

__version__ = "0.3.0"
