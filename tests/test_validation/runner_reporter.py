"""Reporter for dFC API conformance results."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Short display names for the 6 sub-checks (column headers)
_SUBCHECK_LABELS = {
    "registry_instantiation": "instantiation",
    "estimate_FCS_returns_self": "FCS→self",
    "estimate_dFC_returns_DFC": "dFC→DFC",
    "dfc_mat_shape": "shape",
    "symmetry": "symmetric",
    "finite_values": "finite",
}


def _format_table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    sep = "-+-".join("-" * w for w in widths)
    fmt = lambda row: " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row))
    lines = [fmt(headers), sep] + [fmt(r) for r in rows]
    return "\n".join(lines)


class Reporter:
    """Print and save API conformance results."""

    def __init__(self, output_dir: str = "./validation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def print_summary(self, api_conformance: Dict) -> None:
        """Print a table of all methods × sub-checks, then list any errors."""
        if not api_conformance:
            print("No results to report.")
            return

        # Collect ordered sub-check names from the first result
        first = next(iter(api_conformance.values()))
        subcheck_names = [r.name for r in first]
        col_headers = [_SUBCHECK_LABELS.get(n, n) for n in subcheck_names]

        headers = ["Method"] + col_headers + ["TOTAL"]
        rows = []
        failures = []  # (method, subcheck_name, error)

        for method, sub_results in api_conformance.items():
            n_pass = sum(r.passed for r in sub_results)
            n_total = len(sub_results)
            cells = []
            for r in sub_results:
                cells.append("PASS" if r.passed else "FAIL")
                if not r.passed:
                    failures.append((method, r.name, r.error or ""))
            rows.append([method] + cells + [f"{n_pass}/{n_total}"])

        print("\n" + "=" * 80)
        print("API CONFORMANCE RESULTS")
        print("=" * 80)
        print(_format_table(headers, rows))

        if failures:
            print("\nFailure details:")
            for method, name, error in failures:
                err_str = f" — {error}" if error else ""
                print(f"  {method}  |  {name}{err_str}")

    def save_json(self, api_conformance: Dict) -> str:
        """Save conformance results as JSON. Returns the output file path."""
        results_dict = {
            "timestamp": self.timestamp,
            "api_conformance": {
                method_name: [
                    {"name": r.name, "passed": r.passed, "error": r.error}
                    for r in sub_results
                ]
                for method_name, sub_results in api_conformance.items()
            },
        }
        output_file = self.output_dir / f"validation_results_{self.timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=2)
        print(f"\nResults saved to: {output_file}")
        return str(output_file)

    def generate_report(self, api_conformance: Dict, save_json: bool = True) -> None:
        """Print summary table and optionally save to JSON."""
        self.print_summary(api_conformance)
        if save_json:
            self.save_json(api_conformance)
