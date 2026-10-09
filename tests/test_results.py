"""Check complete published results and prove probability corruption is rejected."""
import csv
import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('verify_v2', ROOT / 'verify_v2.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)

class PublishedResults(unittest.TestCase):
    def test_all_saved_results(self):
        result = verifier.verify(results_only=True)
        self.assertTrue(result['passed'])
        self.assertEqual(result['pairs'], 42687)
        self.assertEqual(result['author_pair_decision_disagreements'], {'0.25':0, '0.5':0, '0.9':0})

    def test_probability_corruption_is_rejected(self):
        temp_root = ROOT / 'work/test-tmp'
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            out = Path(directory)
            for path in (ROOT / 'results/v2').iterdir():
                if path.suffix in ('.json', '.csv'): shutil.copy2(path, out / path.name)
            path = out / 'test_pair_predictions.csv'
            with path.open(encoding='utf-8-sig', newline='') as f: rows = list(csv.DictReader(f))
            # Change a probability across the decision boundary but leave all other
            # recorded results untouched. This must invalidate the saved evidence.
            rows[0]['p_killed'] = str(0.0 if float(rows[0]['p_killed']) > .5 else 1.0)
            with path.open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader(); writer.writerows(rows)
            with self.assertRaises(ValueError): verifier.verify(out, results_only=True)

if __name__ == '__main__': unittest.main()
