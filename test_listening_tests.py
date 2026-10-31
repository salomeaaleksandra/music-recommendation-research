import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from listening_tests import build_test_banks, duplicate_groups


class ListeningTests(unittest.TestCase):
    def test_four_banks_sampling_privacy_and_reproducibility(self):
        rng = np.random.default_rng(1)
        values = np.concatenate([rng.normal(c * 8, 1, (80, 3)) for c in range(3)])
        ids = np.arange(len(values))
        labels = np.repeat(np.arange(3), 80)
        audio = dict.fromkeys(ids, 'unused.wav')
        groups = ids.copy()
        groups[1] = groups[0]
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / 'first', Path(temp) / 'second'
            for dest in (first, second):
                manifest, coverage = build_test_banks(values, ids, {'KMeans': labels, 'GaussianMixture': labels}, audio, dest, groups=groups)
            self.assertEqual(len(manifest), 4)
            self.assertTrue((coverage.reason == 'ok').all())
            for model in ['KMeans', 'GaussianMixture']:
                for task in ['ranking', 'ab']:
                    folder = first / model / task
                    public = json.loads((folder / 'participant_tasks.json').read_text())
                    self.assertEqual(public, json.loads((second / model / task / 'participant_tasks.json').read_text()))
                    self.assertTrue(all(set(row) == ({'task_id', 'reference_track_id', 'candidate_track_ids'} if task == 'ranking' else {'task_id', 'reference_track_id', 'a_track_id', 'b_track_id'}) for row in public))
                    private = pd.read_csv(folder / 'private_design.csv')
                    if task == 'ranking':
                        for _, cluster in private.groupby('cluster'):
                            self.assertEqual(cluster.candidate_track_id.nunique(), 10)
                            for _, panel in cluster.groupby('panel'):
                                self.assertTrue(np.all(np.diff(panel.distance) >= panel.minimum_gap.iloc[0]))
                                self.assertEqual(len(set(groups[panel.candidate_track_id]) | {groups[panel.reference_track_id.iloc[0]]}), 6)
                    else:
                        for row in private.itertuples():
                            self.assertEqual(labels[row.within_track_id], labels[row.reference_track_id])
                            self.assertNotEqual(labels[row.outside_track_id], labels[row.reference_track_id])
                            self.assertEqual(len({groups[row.reference_track_id], groups[row.within_track_id], groups[row.outside_track_id]}), 3)

    def test_tied_distances_are_reported_as_uncovered(self):
        with tempfile.TemporaryDirectory() as temp:
            _, coverage = build_test_banks(np.zeros((20, 2)), np.arange(20), {'model': np.zeros(20, dtype=int)}, dict.fromkeys(range(20)), temp)
            ranking = coverage[coverage.task == 'ranking'].iloc[0]
            self.assertEqual(ranking.generated, 0)

    def test_duplicate_identity_is_transitive(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'audio.wav'
            path.write_bytes(b'same recording')
            metadata = pd.DataFrame({('artist', 'name'): ['Artist', 'artist', 'different'], ('track', 'title'): ['Song', 'song', 'other']}, index=[0, 1, 2])
            groups = duplicate_groups(np.arange(3), np.arange(6).reshape(3, 2), {1: path, 2: path}, metadata)
            self.assertEqual(len(set(groups)), 1)


if __name__ == '__main__':
    unittest.main()
