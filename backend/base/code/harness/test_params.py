#!/usr/bin/env python3
"""Tests for durable result and reproducible artifact path separation."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from params import ARTIFACT_ROOT_ENV, InstanceParams


class InstanceParamsStorageTest(unittest.TestCase):
    def test_defaults_to_repository_when_unconfigured(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(
            os.environ, {}, clear=False
        ):
            os.environ.pop(ARTIFACT_ROOT_ENV, None)
            params = InstanceParams(1, rootdir=root)

            self.assertEqual(params.iodir(), Path(root) / "io" / "small")
            self.assertEqual(
                params.measuredir(), Path(root) / "measurements" / "small"
            )

    def test_local_config_moves_only_reproducible_artifacts(self):
        with (
            tempfile.TemporaryDirectory() as root,
            tempfile.TemporaryDirectory() as scratch,
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop(ARTIFACT_ROOT_ENV, None)
            (Path(root) / ".cryptoface-artifact-root").write_text(
                scratch, encoding="utf-8"
            )
            params = InstanceParams(1, rootdir=root)

            self.assertEqual(params.iodir(), Path(scratch) / "io" / "small")
            self.assertEqual(params.dataset_store().parent, Path(scratch) / "datasets")
            self.assertEqual(
                params.measuredir(), Path(root) / "measurements" / "small"
            )

    def test_environment_overrides_local_config(self):
        with (
            tempfile.TemporaryDirectory() as root,
            tempfile.TemporaryDirectory() as configured_scratch,
            tempfile.TemporaryDirectory() as environment_scratch,
            patch.dict(
                os.environ,
                {ARTIFACT_ROOT_ENV: environment_scratch},
                clear=False,
            ),
        ):
            (Path(root) / ".cryptoface-artifact-root").write_text(
                configured_scratch, encoding="utf-8"
            )
            params = InstanceParams(1, rootdir=root)

            self.assertEqual(params.artifact_root, Path(environment_scratch))
            self.assertEqual(
                params.measuredir(), Path(root) / "measurements" / "small"
            )


if __name__ == "__main__":
    unittest.main()
