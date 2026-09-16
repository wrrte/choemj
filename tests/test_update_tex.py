"""Regression checks for the two independent result-to-LaTeX scripts."""
import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (ROOT / "STORM/update_tex.py", ROOT / "Drama/results/update_tex.py")


def load_script(path):
    spec = importlib.util.spec_from_file_location(path.parent.name + "_update_tex", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULES = tuple(load_script(path) for path in SCRIPTS)
LABEL = r"\label{tab:main_performance}" + "\n"
HEADER = r"Game & Random & Human & STORM & STORM+ours & $\Delta$ & DRAMA & DRAMA+ours & $\Delta$"


def row(label, cells):
    return label + " & " + " & ".join(cells) + r" \\ % keep this comment" + "\n"


def table(label):
    lines = [
        "\\begin{table}\n", "\\label{" + label + "}\n",
        "\\begin{tabular}{lrrrrrrrr}\n",
        HEADER + r" \\" + "\n", "\\midrule\n",
    ]
    for game in ("Alien", "Amidar", "Assault", "Asterix", "Freeway"):
        lines.append(row(game, [
            "0", "100", "999", r"\textcolor{blue}{\textbf{999}}", "999",
            "777", r"\textcolor{red}{\textbf{777}}", "777",
        ]))
    lines.append("\\midrule\n")
    for metric in (r"\#Superhuman", "Mean", "Median", "IQM", "Optimality Gap"):
        lines.append(row(metric, ["0", "1", "999", "999", "999", "777", "777", "777"]))
    lines.extend(["\\bottomrule\n", "\\end{tabular}\n", "\\end{table}\n"])
    return "".join(lines)


DOCUMENT = table("tab:before") + table("tab:main_performance") + table("tab:after")
DISPLAY_RESULTS = {
    "Alien": ("50", "80"),
    "Amidar": ("100", "150"),
    "Assault": ("150", "150"),
    "Asterix": ("200", "180"),
}
RESULTS = {
    game: tuple([float(value)] for value in values)
    for game, values in DISPLAY_RESULTS.items()
}


def cells(document):
    body = document.split(LABEL, 1)[1].split(r"\bottomrule", 1)[0]
    return {
        parts[0].strip(): [part.strip() for part in parts]
        for line in body.splitlines() if "&" in line
        for parts in [line.split(r"\\", 1)[0].split("&")]
    }


class UpdateTexTests(unittest.TestCase):
    def test_scores_deltas_and_metrics_use_current_results(self):
        expected_metrics = {
            r"\#Superhuman": ("2", "3", r"\textcolor{green}{+1}"),
            "Mean": ("1.250", "1.400", r"\textcolor{green}{+0.150}"),
            "Median": ("1.250", "1.500", r"\textcolor{green}{+0.250}"),
            "IQM": ("1.250", "1.500", r"\textcolor{green}{+0.250}"),
            "Optimality Gap": ("0.125", "0.050", r"\textcolor{green}{-0.075}"),
        }
        for module in MODULES:
            with self.subTest(module=module.__name__):
                output = "".join(module.update_table(DOCUMENT.splitlines(True), RESULTS))
                updated = cells(output)
                columns = (module.BASE_COLUMN, module.OURS_COLUMN, module.DELTA_COLUMN)
                for game, values in DISPLAY_RESULTS.items():
                    self.assertEqual(tuple(updated[game][col] for col in columns[:2]), values)
                for game, delta in {
                    "Alien": r"\textcolor{green}{+30}",
                    "Amidar": r"\textcolor{green}{+50}",
                    "Assault": "0",
                    "Asterix": r"\textcolor{red}{-20}",
                    "Freeway": "-",
                }.items():
                    self.assertEqual(updated[game][module.DELTA_COLUMN], delta)
                for metric, expected in expected_metrics.items():
                    self.assertEqual(tuple(updated[metric][col] for col in columns), expected)
                self.assertEqual(tuple(updated["Freeway"][col] for col in columns), ("-", "-", "-"))

                # Neither an unrelated table nor the other method may change.
                self.assertEqual(output.split(LABEL)[0], DOCUMENT.split(LABEL)[0])
                self.assertEqual(output.split(r"\bottomrule", 2)[2], DOCUMENT.split(r"\bottomrule", 2)[2])
                before = cells(DOCUMENT)
                for label, values in updated.items():
                    for col in range(9):
                        if col not in columns:
                            self.assertEqual(values[col], before[label][col])
                self.assertEqual(output.count(r"\\ % keep this comment"), DOCUMENT.count(r"\\ % keep this comment"))

    def test_no_results_clears_own_scores_metrics_and_deltas(self):
        for module in MODULES:
            with self.subTest(module=module.__name__):
                updated = cells("".join(module.update_table(DOCUMENT.splitlines(True), {})))
                for label, values in updated.items():
                    if label != "Game":
                        self.assertEqual(
                            [values[col] for col in (module.BASE_COLUMN, module.OURS_COLUMN, module.DELTA_COLUMN)],
                            ["-", "-", "-"],
                        )

    def test_execution_order_and_repeated_updates_are_stable(self):
        storm, drama = MODULES
        original = DOCUMENT.splitlines(True)
        first = drama.update_table(storm.update_table(original, RESULTS), RESULTS)
        second = storm.update_table(drama.update_table(original, RESULTS), RESULTS)
        self.assertEqual(first, second)
        self.assertEqual(first, drama.update_table(storm.update_table(first, RESULTS), RESULTS))
        self.assertEqual("".join(original), DOCUMENT)

    def test_formatting_removes_old_styles_without_a_baseline(self):
        for module in MODULES:
            with self.subTest(module=module.__name__):
                lines = DOCUMENT.splitlines(True)
                index = next(i for i, line in enumerate(lines) if line.startswith("Alien") and i > lines.index(LABEL))
                parts = lines[index].split("&")
                parts[module.BASE_COLUMN] = " - "
                parts[module.OURS_COLUMN] = r" \textcolor{green!50!black}{\textbf{12.5}} "
                lines[index] = "&".join(parts)
                updated = cells("".join(module.format_values(lines)))
                self.assertEqual(updated["Alien"][module.OURS_COLUMN], "12.5")
                self.assertEqual(updated["Alien"][module.DELTA_COLUMN], "-")

    def test_delta_handles_negative_scores_zero_and_decimal_precision(self):
        for module in MODULES:
            with self.subTest(module=module.__name__):
                self.assertEqual(module.format_delta("-20", "-15", None), r"\textcolor{green}{+5}")
                self.assertEqual(module.format_delta("0", "-2", None), r"\textcolor{red}{-2}")
                self.assertEqual(module.format_delta("0", "2", None), r"\textcolor{green}{+2}")
                self.assertEqual(module.format_delta("0.1", "0.3", None), r"\textcolor{green}{+0.2}")
                self.assertEqual(module.format_delta("0.500", "0.500", "Mean"), "0.000")
                self.assertEqual(module.format_delta(None, "1", None), "-")
                self.assertEqual(
                    module.format_delta("0.514", "0.500", "Optimality Gap"),
                    r"\textcolor{green}{-0.014}",
                )
                self.assertEqual(
                    module.format_delta("0.500", "0.514", "Optimality Gap"),
                    r"\textcolor{red}{+0.014}",
                )
                self.assertEqual(module.format_delta("0.500", "0.500", "Optimality Gap"), "0.000")
                self.assertEqual(module.format_delta(None, "0.500", "Optimality Gap"), "-")

    def test_iqm_pools_unrounded_seeds_after_game_specific_normalization(self):
        # Five samples with uneven seed counts. The retained HNS values are
        # [0.02, 0.03, 0.4] and [0.03, 0.04, 0.8], not two game means.
        document = DOCUMENT.replace("Alien & 0 & 100 &", "Alien & 0 & 1 &")
        document = document.replace("Amidar & 0 & 100 &", "Amidar & 100 & 1100 &")
        results = {
            "Alien": ([0.01, 0.02, 0.03, 0.44], [0.02, 0.03, 0.04, 0.85]),
            "Amidar": ([500.0], [900.0]),
            "NotInTable": ([1e12], [2e12]),
        }
        for module in MODULES:
            with self.subTest(module=module.__name__):
                updated = cells("".join(module.update_table(document.splitlines(True), results)))
                self.assertEqual(updated["Alien"][module.BASE_COLUMN], "0.1")
                self.assertEqual(updated["Alien"][module.OURS_COLUMN], "0.2")
                self.assertEqual(updated["IQM"][module.BASE_COLUMN], "0.150")
                self.assertEqual(updated["IQM"][module.OURS_COLUMN], "0.290")
                self.assertEqual(updated["IQM"][module.DELTA_COLUMN], r"\textcolor{green}{+0.140}")
                # Other summaries still use the displayed game means.
                self.assertEqual(updated["Mean"][module.BASE_COLUMN], "0.250")
                self.assertEqual(updated["Mean"][module.OURS_COLUMN], "0.500")

    def test_iqm_does_not_use_existing_table_means_without_seed_results(self):
        for module in MODULES:
            with self.subTest(module=module.__name__):
                updated = cells("".join(module.update_table(DOCUMENT.splitlines(True), {}, reset=False)))
                self.assertNotEqual(updated["Mean"][module.BASE_COLUMN], "-")
                for column in (module.BASE_COLUMN, module.OURS_COLUMN, module.DELTA_COLUMN):
                    self.assertEqual(updated["IQM"][column], "-")

    def test_cli_pairs_seeds_and_uses_latest_duplicate_result(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for script, module in zip(SCRIPTS, MODULES):
                with self.subTest(module=module.__name__):
                    excel = directory / "results.xlsx"
                    tex = directory / "paper.tex"
                    if module.BASE_COLUMN == 3:
                        frame = pd.DataFrame({
                            "Game": ["Alien", None, "Amidar", None],
                            "Config": ["Retrieval 미사용", "target: 16 (anchor 미설정)"] * 2,
                            1: ["10, 999", "30, 888", 100, 200],
                            2: [100, None, None, None],
                            3: [30, 50, 1000, 1200],
                        })
                    else:
                        frame = pd.DataFrame({
                            "Game": ["Alien", "Alien", "Amidar", "Amidar"],
                            "Retrieval": ["X", "O"] * 2,
                            1: ["10, 999", "30, 888", 100, 200],
                            2: [100, None, None, None],
                            3: [30, 50, 1000, 1200],
                        })
                    frame.to_excel(excel, sheet_name="Results", index=False)
                    tex.write_text(DOCUMENT, encoding="utf-8")
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(module.load_results(excel), {
                            "Alien": ([10.0, 30.0], [30.0, 50.0]),
                            "Amidar": ([100.0, 1000.0], [200.0, 1200.0]),
                        })
                    subprocess.run(
                        [sys.executable, str(script), "--excel", str(excel), "--tex", str(tex)],
                        cwd=directory, check=True, capture_output=True, text=True,
                    )
                    updated = cells(tex.read_text(encoding="utf-8"))
                    self.assertEqual(updated["Mean"][module.BASE_COLUMN], "2.850")
                    self.assertEqual(updated["Mean"][module.OURS_COLUMN], "3.700")
                    self.assertEqual(updated["IQM"][module.BASE_COLUMN], "0.650")
                    self.assertEqual(updated["IQM"][module.OURS_COLUMN], "1.250")
                    self.assertEqual(updated["IQM"][module.DELTA_COLUMN], r"\textcolor{green}{+0.600}")
                    self.assertEqual(updated["Alien"][module.DELTA_COLUMN], r"\textcolor{green}{+20}")


if __name__ == "__main__":
    unittest.main()
