import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    'recovery', Path(__file__).with_name('b300_recover_diagnostic.py')
)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


class RecoveryTest(unittest.TestCase):
    def run_recovery(self, *, elapsed=10, user='sa-gha-runner'):
        row = (
            f'2168|b300-dsxe_00|{user}|2026-09-12T08:58:43|FAILED|'
            f'{elapsed}|cpu=192,gres/gpu=8|cpu=192,gres/gpu=8\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'powerx3040').mkdir()
            with patch.dict(os.environ, RUNNER_TEMP=tmp), patch.object(
                recovery.subprocess, 'check_output',
                side_effect=lambda args, **kw: row if args[0] == 'sacct' else '',
            ), patch.object(recovery.subprocess, 'run') as mutation:
                try:
                    recovery.main()
                finally:
                    mutation.assert_not_called()

    def test_reconciled_terminal_job_allows_recovery(self):
        self.run_recovery()

    def test_additional_usage_stops_recovery(self):
        with self.assertRaisesRegex(RuntimeError, 'reconciled'):
            self.run_recovery(elapsed=11)

    def test_foreign_identity_stops_without_cleanup(self):
        with self.assertRaisesRegex(RuntimeError, 'identity'):
            self.run_recovery(user='another-task')


if __name__ == '__main__':
    unittest.main()
