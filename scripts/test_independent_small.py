"""Scheduling metadata tests only; no synthetic inference inputs."""
import copy
import unittest
from run_independent_small import validate_local_gate


class IndependentSmallTests(unittest.TestCase):
    def setUp(self):
        self.prepared = dict(release_sha256='release', native={'binary': 'hash'},
                             dataset_sha256='dataset', settings={'gpu': 'on'})
        self.validation = dict(completed=True, size=0, matching_diagnostic=True,
                               repetitions=1, prepared=copy.deepcopy(self.prepared),
                               matching_validation=dict(passed=True, pairs=1,
                                                        absolute_tolerance=1e-6))

    def test_small_accepts_local_correctness_without_official_single(self):
        validate_local_gate(self.validation, self.prepared, 1)

    def test_no_other_size_is_exempt(self):
        for size in (0, 2, 3):
            with self.assertRaises(ValueError):
                validate_local_gate(self.validation, self.prepared, size)

    def test_failed_incomplete_or_weakened_diagnostic_rejected(self):
        for field, value in (('completed', False), ('matching_diagnostic', False),
                             ('size', 1), ('repetitions', 3)):
            bad = copy.deepcopy(self.validation)
            bad[field] = value
            with self.assertRaises(ValueError):
                validate_local_gate(bad, self.prepared, 1)
        for field, value in (('passed', False), ('pairs', 0), ('absolute_tolerance', 1e-3)):
            bad = copy.deepcopy(self.validation)
            bad['matching_validation'][field] = value
            with self.assertRaises(ValueError):
                validate_local_gate(bad, self.prepared, 1)

    def test_every_identity_must_match(self):
        for field in self.prepared:
            bad = copy.deepcopy(self.validation)
            bad['prepared'][field] = 'changed'
            with self.assertRaises(ValueError):
                validate_local_gate(bad, self.prepared, 1)


if __name__ == '__main__':
    unittest.main()
