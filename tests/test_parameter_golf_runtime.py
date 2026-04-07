from __future__ import annotations

import unittest

from arc.tasks.parameter_golf.runtime import should_use_flash3


class ParameterGolfRuntimeTests(unittest.TestCase):
    def test_h100_gpu_types_enable_flash3(self) -> None:
        self.assertTrue(should_use_flash3("H100"))
        self.assertTrue(should_use_flash3("H100:8"))

    def test_other_gpu_types_do_not_enable_flash3(self) -> None:
        self.assertFalse(should_use_flash3("A100-40GB"))
        self.assertFalse(should_use_flash3("L40S"))
        self.assertFalse(should_use_flash3(None))


if __name__ == "__main__":
    unittest.main()
