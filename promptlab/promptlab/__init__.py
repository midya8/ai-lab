"""promptlab: prompt versioning, paired A/B comparison and regression gates."""
from .template import Template, TemplateError
from .registry import Registry, RegistryError, Version
from .checks import run_check, CheckResult
from .runner import Case, CaseResult, RunResult, load_cases, run_prompt
from .compare import compare_runs, check_regression, render_markdown, Comparison, RegressionReport
from .stats import wilson, mcnemar_exact, paired_bootstrap
from .models import RuleModel, CommandModel, make_model

__version__ = "0.1.0"
