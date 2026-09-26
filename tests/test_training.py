import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

class TrainingTests(unittest.TestCase):
    def test_real_adapter_training_and_resume(self):
        self.assertIsNotNone(importlib.util.find_spec('loraforge.smoke'),'Real training smoke test must be implemented')
        from loraforge.smoke import run_smoke
        with tempfile.TemporaryDirectory() as d:
            r=run_smoke(Path(d))
            self.assertTrue(r['passed'])
            self.assertTrue(r['base_weights_unchanged'])
            self.assertTrue(r['resume_matches_uninterrupted'])
            self.assertTrue(r['reload_matches'])
            self.assertTrue(r['adapter_changed'])

if __name__=='__main__': unittest.main()
