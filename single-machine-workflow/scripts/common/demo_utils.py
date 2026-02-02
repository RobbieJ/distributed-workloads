"""Shared utilities for demo script output validation and ASCII charts.

Provides MetricsCollector for accumulating training metrics,
OutputValidator for pass/fail checks, and chart functions using plotext.
"""

import json
import logging
import os
import time
import warnings


# ── Intro Banner ───────────────────────────────────────────────────────────


def print_intro(title, what, why, watch_for):
    """Print a narrative banner explaining the demo."""
    BOLD = "\033[1m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    width = 64
    print()
    print(f"{CYAN}{'━' * width}{RESET}")
    print(f"{CYAN}┃{RESET} {BOLD}{title}{RESET}")
    print(f"{CYAN}{'━' * width}{RESET}")
    print(f"{CYAN}┃{RESET} {BOLD}WHAT:{RESET}  {what}")
    print(f"{CYAN}┃{RESET} {BOLD}WHY:{RESET}   {why}")
    print(f"{CYAN}┃{RESET} {BOLD}WATCH:{RESET} {watch_for}")
    print(f"{CYAN}{'━' * width}{RESET}")
    print()


# ── Warning Suppression ───────────────────────────────────────────────────


def suppress_known_warnings():
    """Suppress noisy warnings that distract from demo output."""
    # CUDA capability warning (GB10)
    warnings.filterwarnings("ignore", message=".*CUDA capability.*")
    # Tokenizer class deprecated warnings
    warnings.filterwarnings("ignore", message=".*class.*Tokenizer.*deprecated.*")
    # Transformers FutureWarning spam
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
    # Set transformers logger to ERROR level
    logging.getLogger("transformers").setLevel(logging.ERROR)


# ── MetricsCollector ────────────────────────────────────────────────────────


class MetricsCollector:
    """Accumulates step/epoch/trial metrics and saves to JSON."""

    def __init__(self, script_name):
        self.script_name = script_name
        self.steps = []
        self.epochs = []
        self.trials = []
        self.metadata = {}
        self.start_time = time.time()

    def add_step(self, step, **kwargs):
        entry = {"step": step, **kwargs}
        self.steps.append(entry)

    def add_epoch(self, epoch, **kwargs):
        entry = {"epoch": epoch, **kwargs}
        self.epochs.append(entry)

    def add_trial(self, trial_number, **kwargs):
        entry = {"trial": trial_number, **kwargs}
        self.trials.append(entry)

    def set_metadata(self, **kwargs):
        self.metadata.update(kwargs)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "script": self.script_name,
            "elapsed_seconds": round(time.time() - self.start_time, 1),
            "metadata": self.metadata,
        }
        if self.steps:
            data["steps"] = self.steps
        if self.epochs:
            data["epochs"] = self.epochs
        if self.trials:
            data["trials"] = self.trials
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Metrics saved to {path}")


# ── OutputValidator ─────────────────────────────────────────────────────────


class OutputValidator:
    """Runs checks and prints a colored PASS/FAIL terminal report."""

    def __init__(self, title, test_mode=False):
        self.title = title
        self.test_mode = test_mode
        self.results = []  # list of (status, description); status: True/False/"warn"

    def _record(self, passed, description):
        self.results.append((passed, description))
        return passed

    def check_or_warn(self, condition, message, test_mode_note=""):
        """Like check(), but downgrades FAIL to WARN in test mode."""
        if condition:
            self.results.append((True, message))
        elif self.test_mode:
            note = f" [test-mode: {test_mode_note}]" if test_mode_note else ""
            self.results.append(("warn", f"{message}{note}"))
        else:
            self.results.append((False, message))
        return condition

    def check_dir_exists(self, path, description="Directory exists"):
        return self._record(
            os.path.isdir(path) and len(os.listdir(path)) > 0,
            description,
        )

    def check_file_exists(self, path, description="File exists"):
        return self._record(os.path.isfile(path), description)

    def check_file_min_size(self, path, min_bytes, description="File size check"):
        try:
            size = os.path.getsize(path)
            return self._record(size >= min_bytes, description)
        except OSError:
            return self._record(False, description)

    def check_metric_range(self, value, lo, hi, description="Metric in range"):
        try:
            return self._record(lo <= float(value) <= hi, description)
        except (TypeError, ValueError):
            return self._record(False, description)

    def check_loss_decreased(self, first, last, description="Loss decreased"):
        try:
            return self._record(float(last) < float(first), description)
        except (TypeError, ValueError):
            return self._record(False, description)

    def check_callable(self, fn, description="Callable check"):
        try:
            fn()
            return self._record(True, description)
        except Exception:
            return self._record(False, description)

    def check(self, condition, description="Custom check"):
        return self._record(bool(condition), description)

    def print_report(self):
        """Print formatted table and return True if all passed (warns don't count as failures)."""
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        BOLD = "\033[1m"
        RESET = "\033[0m"

        print(f"\n{'=' * 60}")
        print(f"{BOLD}Validation Report: {self.title}{RESET}")
        print(f"{'=' * 60}")

        has_failures = False
        for status, desc in self.results:
            if status is True:
                label = f"{GREEN}PASS{RESET}"
            elif status == "warn":
                label = f"{YELLOW}WARN{RESET}"
            else:
                label = f"{RED}FAIL{RESET}"
                has_failures = True
            print(f"  [{label}] {desc}")

        total = len(self.results)
        passed_count = sum(1 for s, _ in self.results if s is True)
        warn_count = sum(1 for s, _ in self.results if s == "warn")
        fail_count = total - passed_count - warn_count
        summary_color = GREEN if not has_failures else RED
        summary = f"{passed_count}/{total} checks passed"
        if warn_count:
            summary += f", {warn_count} warnings"
        print(f"\n{summary_color}{summary}{RESET}")
        print(f"{'=' * 60}\n")
        return not has_failures


# ── Chart Functions ─────────────────────────────────────────────────────────

_PLOTEXT_AVAILABLE = None


def _has_plotext():
    global _PLOTEXT_AVAILABLE
    if _PLOTEXT_AVAILABLE is None:
        try:
            import plotext  # noqa: F401

            _PLOTEXT_AVAILABLE = True
        except ImportError:
            _PLOTEXT_AVAILABLE = False
    return _PLOTEXT_AVAILABLE


def plot_loss_curve(steps, losses, title="Training Loss"):
    """Render a line chart of loss over steps."""
    if not _has_plotext() or not steps:
        return
    import plotext as plt

    plt.clear_figure()
    plt.plot(steps, losses, marker="braille")
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.show()


def plot_multi_loss(series_dict, title="Metrics", xlabel="Step"):
    """Render multiple series on one chart.

    series_dict: {"label": (x_list, y_list), ...}
    """
    if not _has_plotext() or not series_dict:
        return
    import plotext as plt

    plt.clear_figure()
    for label, (xs, ys) in series_dict.items():
        if xs and ys:
            plt.plot(xs, ys, label=label, marker="braille")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Value")
    plt.show()


def plot_bar_chart(labels, values, title="Bar Chart"):
    """Render a horizontal bar chart."""
    if not _has_plotext() or not labels:
        return
    import plotext as plt

    plt.clear_figure()
    plt.bar(labels, values)
    plt.title(title)
    plt.show()


# ── BaselineComparator ──────────────────────────────────────────────────────


class BaselineComparator:
    """Collects baseline vs technique metrics and renders a comparison table."""

    def __init__(self, title):
        self.title = title
        self.metrics = []  # list of dicts with name/baseline/technique/unit/lower_is_better

    def add_metric(self, name, baseline, technique, unit="", lower_is_better=True):
        self.metrics.append({
            "name": name,
            "baseline": baseline,
            "technique": technique,
            "unit": unit,
            "lower_is_better": lower_is_better,
        })

    def render(self):
        """Print a colored comparison table with improvement percentages."""
        GREEN = "\033[92m"
        RED = "\033[91m"
        BOLD = "\033[1m"
        RESET = "\033[0m"

        print(f"\n{'=' * 60}")
        print(f"{BOLD}Baseline vs Technique: {self.title}{RESET}")
        print(f"{'=' * 60}")

        headers = ["Metric", "Baseline", "Technique", "Change"]
        rows = []
        for m in self.metrics:
            b = m["baseline"]
            t = m["technique"]
            unit = m["unit"]

            b_str = f"{b:.4f}{unit}" if isinstance(b, float) else f"{b}{unit}"
            t_str = f"{t:.4f}{unit}" if isinstance(t, float) else f"{t}{unit}"

            # Compute percentage change if both are numeric
            try:
                b_num = float(b)
                t_num = float(t)
                if b_num != 0:
                    pct = ((t_num - b_num) / abs(b_num)) * 100
                    is_better = (pct < 0) if m["lower_is_better"] else (pct > 0)
                    color = GREEN if is_better else RED
                    change_str = f"{color}{pct:+.1f}%{RESET}"
                else:
                    change_str = "N/A"
            except (TypeError, ValueError):
                change_str = ""

            rows.append([m["name"], b_str, t_str, change_str])

        plot_table(headers, rows, title=None)

    def to_dict(self):
        """Return serializable dict for metrics.json."""
        entries = []
        for m in self.metrics:
            entry = {
                "name": m["name"],
                "baseline": m["baseline"],
                "technique": m["technique"],
            }
            if m["unit"]:
                entry["unit"] = m["unit"]
            # Compute improvement percentage
            try:
                b_num = float(m["baseline"])
                t_num = float(m["technique"])
                if b_num != 0:
                    pct = ((t_num - b_num) / abs(b_num)) * 100
                    entry["change_pct"] = round(pct, 1)
            except (TypeError, ValueError):
                pass
            entries.append(entry)
        return {"title": self.title, "metrics": entries}

    def add_validations(self, validator):
        """Add checks that technique beat baseline on key metrics."""
        for m in self.metrics:
            try:
                b_num = float(m["baseline"])
                t_num = float(m["technique"])
                if m["lower_is_better"]:
                    validator.check(
                        t_num <= b_num,
                        f"{m['name']}: technique ({t_num:.4f}) <= baseline ({b_num:.4f})",
                    )
                else:
                    validator.check(
                        t_num >= b_num,
                        f"{m['name']}: technique ({t_num:.4f}) >= baseline ({b_num:.4f})",
                    )
            except (TypeError, ValueError):
                pass


def plot_comparison_bar(labels, baseline_vals, technique_vals, title="Baseline vs Technique"):
    """Render a grouped bar chart showing baseline vs technique side by side."""
    if not _has_plotext() or not labels:
        return
    import plotext as plt

    plt.clear_figure()
    plt.multiple_bar(
        labels,
        [baseline_vals, technique_vals],
        labels=["Baseline", "Technique"],
    )
    plt.title(title)
    plt.show()


def plot_table(headers, rows, title=None):
    """Print a formatted table using tabulate (fallback to manual)."""
    if title:
        print(f"\n{title}")
        print("-" * len(title))
    try:
        from tabulate import tabulate

        print(tabulate(rows, headers=headers, tablefmt="grid"))
    except ImportError:
        # Manual fallback
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
        print(fmt.format(*headers))
        print("  ".join("-" * w for w in col_widths))
        for row in rows:
            print(fmt.format(*[str(c) for c in row]))
    print()
