"""Reproducible task banks; participant files never contain model answers."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def duplicate_groups(ids, values, audio_paths, metadata=None):
    """Conservatively merge exact audio/features and matching artist/title identities."""
    parent = list(range(len(ids)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, track in enumerate(ids):
        keys = [('features', hashlib.sha256(np.ascontiguousarray(values[i]).tobytes()).hexdigest())]
        if int(track) in audio_paths:
            with open(audio_paths[int(track)], 'rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            keys.append(('audio', digest))
        if metadata is not None and track in metadata.index:
            artist = metadata.loc[track, ('artist', 'name')]
            title = metadata.loc[track, ('track', 'title')]
            if pd.notna(artist) and pd.notna(title) and str(artist).strip() and str(title).strip():
                keys.append(('title', str(artist).strip().casefold(), str(title).strip().casefold()))
        for key in keys:
            if key in seen:
                parent[root(i)] = root(seen[key])
            else:
                seen[key] = i
    return np.array([root(i) for i in range(len(ids))])


def build_test_banks(values, ids, labels_by_model, audio_paths, output_dir,
                     groups=None, seed=42, panels=2, candidates=5,
                     min_gap_fraction=0.05, attempts=2000):
    """Create two ranking panels and typical/hard A/B comparisons per cluster.

    Ranking bands are quintiles of the original eligible distance ordering.
    Minimum adjacent gap is a fraction of its 5th–95th percentile distance span.
    Typical A/B draws uniformly outside; hard A/B draws from the nearest other
    empirical centroid. A/B does not impose a feature-distance answer.
    """
    values, ids = np.asarray(values), np.asarray(ids)
    if len(values) != len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError('Expected aligned features and unique track IDs.')
    if not 0 <= min_gap_fraction < 1 or panels < 1 or candidates < 2:
        raise ValueError('Invalid panel or distance-gap configuration.')
    groups = np.arange(len(ids)) if groups is None else np.asarray(groups)
    if len(groups) != len(ids):
        raise ValueError('Duplicate groups must align with track IDs.')
    available = np.array([int(track) in audio_paths for track in ids])
    manifest, coverage = [], []
    output_dir = Path(output_dir)
    for model_number, (name, labels) in enumerate(labels_by_model.items()):
        labels = np.asarray(labels)
        if len(labels) != len(ids) or pd.isna(labels).any():
            raise ValueError(f'Missing or misaligned labels for {name}.')
        rng = np.random.default_rng(np.random.SeedSequence([seed, model_number]))
        clusters = np.unique(labels)
        centers = {c: values[labels == c].mean(axis=0) for c in clusters}
        public = {'ranking': [], 'ab': []}
        private = {'ranking': [], 'ab': []}
        for cluster in clusters:
            members = np.flatnonzero((labels == cluster) & available)
            if not len(members):
                coverage.append(dict(model=name, cluster=int(cluster), task='both', generated=0, reason='no_audio'))
                continue
            distances_to_center = np.linalg.norm(values[members] - centers[cluster], axis=1)
            reference = members[np.lexsort((ids[members], distances_to_center))[0]]
            # One representative per duplicate group, excluding all reference aliases.
            def unique_pool(indices):
                chosen, seen = [], {groups[reference]}
                for i in indices:
                    if groups[i] not in seen:
                        seen.add(groups[i]); chosen.append(i)
                return np.asarray(chosen, dtype=int)

            pool = unique_pool(members)
            distance = np.linalg.norm(values - values[reference], axis=1)
            ordered = pool[np.lexsort((ids[pool], distance[pool]))]
            bands = np.array_split(ordered, candidates)
            span = float(np.ptp(np.quantile(distance[pool], [.05, .95]))) if len(pool) else 0.
            gap = max(min_gap_fraction * span, np.finfo(float).eps)
            accepted = None
            # Sample both disjoint panels jointly; never relax a gap silently.
            if all(len(band) >= panels for band in bands):
                for _ in range(attempts):
                    selection = np.array([rng.choice(band, panels, replace=False) for band in bands]).T
                    if np.all(np.diff(distance[selection], axis=1) >= gap):
                        accepted = selection
                        break
            for panel, selection in enumerate([] if accepted is None else accepted, 1):
                task_id = f'{name}_ranking_{int(cluster)}_{panel}'
                shuffled = rng.permutation(selection)
                public['ranking'].append(dict(task_id=task_id, reference_track_id=int(ids[reference]),
                    candidate_track_ids=ids[shuffled].astype(int).tolist()))
                for rank, i in enumerate(selection, 1):
                    private['ranking'].append(dict(task_id=task_id, cluster=int(cluster), panel=panel,
                        reference_track_id=int(ids[reference]), candidate_track_id=int(ids[i]),
                        model_predicted_rank=rank, distance=float(distance[i]), minimum_gap=gap,
                        display_position=int(np.flatnonzero(shuffled == i)[0]) + 1))
            coverage.append(dict(model=name, cluster=int(cluster), task='ranking',
                generated=0 if accepted is None else panels,
                reason='insufficient_bands_or_gap' if accepted is None else 'ok'))
            other_clusters = [c for c in clusters if c != cluster and np.any((labels == c) & available)]
            nearest = min(other_clusters, key=lambda c: np.linalg.norm(centers[c] - centers[cluster])) if other_clusters else None
            used = set()
            for condition in ['typical', 'hard']:
                within = np.array([i for i in pool if groups[i] not in used], dtype=int)
                outside = unique_pool(np.flatnonzero(available & (labels != cluster) &
                    ((labels == nearest) if condition == 'hard' else True)))
                outside = np.array([i for i in outside if groups[i] not in used], dtype=int)
                if not len(within) or not len(outside):
                    coverage.append(dict(model=name, cluster=int(cluster), task=f'ab_{condition}', generated=0, reason='insufficient_unique_audio'))
                    continue
                a = int(rng.choice(within))
                outside = outside[groups[outside] != groups[a]]
                if not len(outside):
                    coverage.append(dict(model=name, cluster=int(cluster), task=f'ab_{condition}', generated=0, reason='duplicate_only_outside'))
                    continue
                b = int(rng.choice(outside))
                used.update([groups[a], groups[b]])
                pair = rng.permutation([a, b])
                task_id = f'{name}_ab_{int(cluster)}_{condition}'
                public['ab'].append(dict(task_id=task_id, reference_track_id=int(ids[reference]),
                    a_track_id=int(ids[pair[0]]), b_track_id=int(ids[pair[1]])))
                private['ab'].append(dict(task_id=task_id, cluster=int(cluster), condition=condition,
                    reference_track_id=int(ids[reference]), within_track_id=int(ids[a]), outside_track_id=int(ids[b]),
                    outside_cluster=int(labels[b]), within_position='A' if pair[0] == a else 'B',
                    within_distance=float(distance[a]), outside_distance=float(distance[b])))
                coverage.append(dict(model=name, cluster=int(cluster), task=f'ab_{condition}', generated=1, reason='ok'))
        for task in public:
            folder = output_dir / name / task
            folder.mkdir(parents=True, exist_ok=True)
            rng.shuffle(public[task])
            # Opaque IDs prevent model, cluster and condition leaking to listeners.
            id_map = {row['task_id']: f'task_{i:04d}' for i, row in enumerate(public[task], 1)}
            for row in public[task]:
                row['task_id'] = id_map[row['task_id']]
            for row in private[task]:
                row['task_id'] = id_map[row['task_id']]
            (folder / 'participant_tasks.json').write_text(json.dumps(public[task], indent=2), encoding='utf8')
            pd.DataFrame(private[task]).to_csv(folder / 'private_design.csv', index=False)
            manifest.append(dict(model=name, task=task, tasks=len(public[task]), directory=str(folder)))
    coverage = pd.DataFrame(coverage)
    coverage.to_csv(output_dir / 'coverage.csv', index=False)
    manifest = pd.DataFrame(manifest)
    manifest.to_csv(output_dir / 'manifest.csv', index=False)
    return manifest, coverage
