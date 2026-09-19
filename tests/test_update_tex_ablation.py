"""Check paired-seed aggregation and isolated updates of the ablation table."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'STORM'))
try:
    import update_tex_ablation as ablation
finally:
    sys.path.pop(0)


def document():
    lines = [
        r'\begin{table}',
        r'\label{tab:main_performance}',
        r'\begin{tabular}{lllrrrrrr}',
        r'Game & Random & Human & STORM & STORM+ours & $\Delta$ & DRAMA & DRAMA+ours & $\Delta$ \\',
    ]
    for game in ('Alien', 'Frostbite', 'BankHeist', 'Freeway'):
        lines.append(game + r' & 0 & 100 & 999 & 999 & 0 & 777 & 777 & 0 \\')
    for metric in (r'\#Superhuman', 'Mean', 'Median', 'IQM', 'Optimality Gap'):
        lines.append(metric + r' & 0 & 1 & 999 & 999 & 0 & 777 & 777 & 0 \\')
    lines.extend([
        r'\bottomrule', r'\end{tabular}', r'\end{table}',
        ablation.BEGIN_MARKER, 'old table', ablation.END_MARKER,
        r'\section{Unrelated content}',
    ])
    return '\n'.join(lines) + '\n'


def table_cells(text):
    block = text.split(ablation.BEGIN_MARKER)[1].split(ablation.END_MARKER)[0]
    return {
        parts[0].strip(): [part.strip() for part in parts[1:]]
        for line in block.splitlines() if '&' in line
        for parts in [line.split(r'\\', 1)[0].split('&')]
    }


class AblationTests(unittest.TestCase):
    def test_cli_pairs_target_configs_excludes_seeds_and_preserves_document(self):
        frame = pd.DataFrame({
            'Game': ['Alien\nRandom: 0\nHuman: 100', None, None,
                     'Frostbite', None, 'BankHeist', None, 'Freeway'],
            'Config': [*ablation.CONFIGS, 'Retrieval 미사용',
                       *ablation.CONFIGS, *ablation.CONFIGS, ablation.CONFIGS[0]],
            '1': ['10, 999 (w: 60000)', '30, 888', 9999, 300, 600, 100, 200, 999],
            '2': [100, 'running', 9999, None, None, None, None, None],
            '3': [30, 50, 9999, None, None, None, None, None],
            '10': [None, None, None, 10000, 20000, None, None, None],
            '6020': [None, None, None, None, None, 10000, 20000, None],
            'Mean (공통 시드)': [99999] * 8,
        })
        original = document()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            excel = directory / 'input.xlsx'
            tex = directory / 'paper.tex'
            frame.to_excel(excel, sheet_name='Results', index=False)
            tex.write_text(original, encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                results = ablation.load_results(excel, configs=ablation.CONFIGS)
            self.assertEqual(results, {
                'Alien': ([10.0, 30.0], [30.0, 50.0]),
                'Frostbite': ([300.0], [600.0]),
                'BankHeist': ([100.0], [200.0]),
            })
            subprocess.run(
                [sys.executable, str(ROOT / 'STORM/update_tex_ablation.py'),
                 '--excel', str(excel), '--tex', str(tex)],
                cwd=directory, check=True, capture_output=True, text=True,
            )
            output = tex.read_text(encoding='utf-8')
        self.assertEqual(output.split(ablation.BEGIN_MARKER)[0], original.split(ablation.BEGIN_MARKER)[0])
        self.assertEqual(output.split(ablation.END_MARKER)[1], original.split(ablation.END_MARKER)[1])
        self.assertEqual(ablation.update_ablation_table(output, results), output)
        self.assertIn('3 games and 4 seed pairs', output)
        cells = table_cells(output)
        for game, expected in {
            'Alien': ['2', '20', '40'],
            'Frostbite': ['1', '300', '600'],
            'BankHeist': ['1', '100', '200'],
            'Freeway': ['0', '-', '-'],
        }.items():
            self.assertEqual(cells[game], expected)
        for metric, expected in {
            r'\#Superhuman': ['1', '2'],
            'Mean': ['1.400', '2.800'],
            'Median': ['1.000', '2.000'],
            'IQM': ['0.650', '1.250'],
            'Optimality Gap': ['0.267', '0.200'],
        }.items():
            self.assertEqual(cells[metric], ['', *expected])

    def test_empty_comparison_clears_previous_ablation_values(self):
        original = ablation.update_ablation_table(document(), {'Alien': ([1.0], [2.0])})
        output = ablation.update_ablation_table(original, {})
        self.assertEqual(table_cells(output)['Alien'], ['0', '-', '-'])
        self.assertEqual(table_cells(output)['IQM'], ['', '-', '-'])
        self.assertIn('0 games and 0 seed pairs', output)

    def test_missing_duplicate_or_reversed_markers_are_rejected(self):
        original = document()
        invalid = [
            original.replace(ablation.BEGIN_MARKER, ''),
            original + ablation.BEGIN_MARKER + '\n',
            original.replace(ablation.BEGIN_MARKER, 'TEMP').replace(
                ablation.END_MARKER, ablation.BEGIN_MARKER).replace('TEMP', ablation.END_MARKER),
        ]
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                ablation.update_ablation_table(text, {})


if __name__ == '__main__':
    unittest.main()
