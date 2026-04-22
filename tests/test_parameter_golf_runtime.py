from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from arc.errors import ArcError
from arc.tasks.parameter_golf.runtime import (
    DEFAULT_GPU_TYPE,
    DEFAULT_REMOTE_CPU,
    DEFAULT_REMOTE_MEMORY_GB,
    SUBMIT_GRAD_ACCUM_STEPS,
    SUBMIT_MAX_WALLCLOCK_SECONDS,
    ParameterGolfModalRunner,
    submission_run_root_from_entrypoint,
    should_use_flash3,
)


class ParameterGolfRuntimeTests(unittest.TestCase):
    def _create_repo(self) -> Path:
        root = Path(self.tempdir.name)
        (root / "pyproject.toml").write_text("[project]\nname='task'\nversion='0.0.0'\n", encoding="utf-8")
        (root / "train_gpt.py").write_text("print('train')\n", encoding="utf-8")
        (root / "prepare.py").write_text("print('prepare')\n", encoding="utf-8")
        workspace = root / "workspace"
        workspace.mkdir()
        (workspace / "custom_train.py").write_text("print('custom train')\n", encoding="utf-8")
        return root

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.repo_root = self._create_repo()
        self.runner = ParameterGolfModalRunner(self.repo_root)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_submission_run_root_from_entrypoint(self) -> None:
        self.assertEqual(
            submission_run_root_from_entrypoint(".arc/submissions/20260409-fast-muonplus/train_gpt.py"),
            "20260409-fast-muonplus",
        )
        self.assertIsNone(submission_run_root_from_entrypoint("train_gpt.py"))
        self.assertIsNone(submission_run_root_from_entrypoint(".arc/submissions/foo.py"))
        self.assertIsNone(
            submission_run_root_from_entrypoint(".arc/submissions"),
        )

    def test_run_config_submission_train_sets_submission_outputs_and_run_id(self) -> None:
        sub_root = self.repo_root / ".arc" / "submissions" / "20260409-fast-muonplus"
        sub_root.mkdir(parents=True)
        (sub_root / "train_gpt.py").write_text("print('sub')\n", encoding="utf-8")
        (self.repo_root / ".env").write_text("RUN_ID=should-be-overridden\n", encoding="utf-8")
        with mock.patch.dict(os.environ, {}, clear=True):
            config = self.runner._build_run_config(
                "train",
                [".arc/submissions/20260409-fast-muonplus/train_gpt.py"],
                quiet=False,
            )
        self.assertTrue(config.submission_outputs)
        self.assertEqual(config.run_id, "20260409-fast-muonplus")
        self.assertEqual(config.train_entrypoint, ".arc/submissions/20260409-fast-muonplus/train_gpt.py")
        self.assertEqual(config.forwarded_env.get("RUN_ID"), "should-be-overridden")

    def test_h100_gpu_types_enable_flash3(self) -> None:
        self.assertTrue(should_use_flash3("H100"))
        self.assertTrue(should_use_flash3("H100:8"))

    def test_other_gpu_types_do_not_enable_flash3(self) -> None:
        self.assertFalse(should_use_flash3("A100-40GB"))
        self.assertFalse(should_use_flash3("L40S"))
        self.assertFalse(should_use_flash3(None))

    def test_run_config_allows_overrides_and_forwards_selected_env(self) -> None:
        (self.repo_root / ".env").write_text("TRAIN_BATCH_TOKENS=123456\n", encoding="utf-8")
        with mock.patch.dict(
            os.environ,
            {
                "RUN_ID": "debug-a100",
                "MAX_WALLCLOCK_SECONDS": "45",
            },
            clear=True,
        ):
            config = self.runner._build_run_config(
                "train",
                ["workspace/custom_train.py", "--", "--some-flag", "value"],
                quiet=True,
                gpu="H100:8",
                cpu=16.0,
                memory_gb=64.0,
            )

        self.assertEqual(config.mode, "run")
        self.assertEqual(config.action, "train")
        self.assertTrue(config.quiet)
        self.assertEqual(config.gpu_type, "H100:8")
        self.assertEqual(config.cpu, 16.0)
        self.assertEqual(config.memory_gb, 64.0)
        self.assertEqual(config.train_entrypoint, "workspace/custom_train.py")
        self.assertEqual(config.extra_args, ["--some-flag", "value"])
        self.assertEqual(config.run_id, "debug-a100")
        self.assertTrue(config.use_flash3)
        self.assertEqual(config.forwarded_env["RUN_ID"], "debug-a100")
        self.assertEqual(config.forwarded_env["MAX_WALLCLOCK_SECONDS"], "45")
        self.assertEqual(config.forwarded_env["TRAIN_BATCH_TOKENS"], "123456")
        self.assertFalse(config.submission_outputs)

    def test_submit_config_is_arc_owned_and_hardcodes_wallclock(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "ARC_PARAMETER_GOLF_GPU": "H100:8",
                "ARC_PARAMETER_GOLF_CPU": "16",
                "ARC_PARAMETER_GOLF_MEMORY_GB": "64",
                "RUN_ID": "debug-run",
                "ITERATIONS": "999",
                "MAX_WALLCLOCK_SECONDS": "999",
            },
            clear=True,
        ):
            config = self.runner._build_submit_train_config()

        self.assertEqual(config.mode, "submit")
        self.assertEqual(config.action, "train")
        self.assertFalse(config.quiet)
        self.assertEqual(config.gpu_type, DEFAULT_GPU_TYPE)
        self.assertEqual(config.cpu, DEFAULT_REMOTE_CPU)
        self.assertEqual(config.memory_gb, DEFAULT_REMOTE_MEMORY_GB)
        self.assertIsNone(config.train_entrypoint)
        self.assertEqual(config.extra_args, [])
        self.assertEqual(config.run_id, self.repo_root.name)
        self.assertTrue(config.use_flash3)
        self.assertEqual(
            config.forwarded_env["MAX_WALLCLOCK_SECONDS"],
            str(SUBMIT_MAX_WALLCLOCK_SECONDS),
        )
        self.assertEqual(
            config.forwarded_env["GRAD_ACCUM_STEPS"],
            str(SUBMIT_GRAD_ACCUM_STEPS),
        )
        self.assertNotIn("ITERATIONS", config.forwarded_env)
        self.assertNotIn("RUN_ID", config.forwarded_env)
        self.assertFalse(config.submission_outputs)

    def test_submit_config_respects_train_wallclock_override(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            config = self.runner._build_submit_train_config(train_wallclock=420)

        self.assertEqual(config.forwarded_env["MAX_WALLCLOCK_SECONDS"], "420")

    def test_submit_config_uses_default_wallclock_when_none(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            config = self.runner._build_submit_train_config(train_wallclock=None)

        self.assertEqual(
            config.forwarded_env["MAX_WALLCLOCK_SECONDS"],
            str(SUBMIT_MAX_WALLCLOCK_SECONDS),
        )

    def test_submit_config_respects_grad_accum_override(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            config = self.runner._build_submit_train_config(grad_accum_steps=4)

        self.assertEqual(config.forwarded_env["GRAD_ACCUM_STEPS"], "4")

    def test_submit_config_rejects_invalid_grad_accum(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ArcError):
                self.runner._build_submit_train_config(grad_accum_steps=3)


if __name__ == "__main__":
    unittest.main()
