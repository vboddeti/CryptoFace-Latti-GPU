#!/usr/bin/env python3
"""
params.py - Parameters and directory structure for the submission.
"""
# Copyright 2025 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path


ARTIFACT_ROOT_ENV = "CRYPTOFACE_ARTIFACT_ROOT"
ARTIFACT_ROOT_FILE = ".cryptoface-artifact-root"
DATASET_STORE_NAME = "face_dataset.h5"


def configured_artifact_root(rootdir):
    """Return the root for regenerable artifacts, separate from final results."""
    rootdir = Path(rootdir)
    configured = os.environ.get(ARTIFACT_ROOT_ENV)
    if configured is None:
        config_file = rootdir / ARTIFACT_ROOT_FILE
        if config_file.is_file():
            configured = config_file.read_text(encoding="utf-8").strip()

    if not configured:
        return rootdir

    artifact_root = Path(configured).expanduser()
    if not artifact_root.is_absolute():
        artifact_root = rootdir / artifact_root
    return artifact_root

# Enum for benchmark size
SINGLE = 0
SMALL = 1
MEDIUM = 2
LARGE = 3

def instance_name(size):
    """Return the string name of the instance size."""
    if size > LARGE:
        return "unknown"
    names = ["single", "small", "medium", "large"]
    return names[size]

class InstanceParams:
    """Parameters that differ for different instance sizes."""

    def __init__(self, size, rootdir=None):
        """Constructor."""
        self.size = size
        self.rootdir = Path(rootdir) if rootdir else Path.cwd()
        self.artifact_root = configured_artifact_root(self.rootdir)

        if size > LARGE:
            raise ValueError("Invalid instance size")

        # Face-pair counts for the single / small / medium / large variants.
        batch_sizes = [1, 128, 256, 1024]
        self.batch_size = batch_sizes[size]

    def datadir(self):
        """Return the dataset directory path."""
        return self.artifact_root / "datasets" / instance_name(self.size)

    def dataset_store(self):
        """Return the indexed, reproducible dataset store path."""
        return self.artifact_root / "datasets" / DATASET_STORE_NAME
    
    def dataset_intermediate_dir(self):
        """Return the intermediate directory path."""
        return self.datadir() / "intermediate"

    def iodir(self):
        """Return the I/O directory path."""
        return self.artifact_root / "io" / instance_name(self.size)

    def io_root(self):
        """Return the root for reproducible benchmark I/O."""
        return self.artifact_root / "io"

    def get_server_data_dir(self):
        """Return the shared server data directory path."""
        return self.io_root() / "server_data"

    def io_intermediate_dir(self):
        """Return the intermediate directory path."""
        return self.iodir() / "intermediate"

    def measuredir(self):
        """Return the measurements directory path."""
        return self.rootdir / "measurements" / instance_name(self.size)
    
    def get_batch_size(self):
        """Return the number of items in the batch."""
        return self.batch_size

    def get_test_input_file(self):
        """Return the test input file path."""
        return self.dataset_intermediate_dir() / "test_pairs.h5"

    def get_selection_file(self):
        """Return the compact run-level source-row selection."""
        return self.dataset_intermediate_dir() / "test_selection.npz"

    def get_ground_truth_labels_file(self):
        """Return the ground truth labels file path."""
        return self.dataset_intermediate_dir() / "test_labels.txt"

    def get_encrypted_model_predictions_file(self):
        """Return the encrypted model predictions file path."""
        return self.iodir() / "encrypted_model_predictions.txt"

    def get_harness_model_predictions_file(self):
        """Return the harness model predictions file path."""
        return self.iodir() / "harness_model_predictions.txt"
