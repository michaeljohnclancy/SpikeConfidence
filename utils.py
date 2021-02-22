from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from spikeforest2_utils import AutoRecordingExtractor, AutoSortingExtractor

from sorter import SpikeSession


def prepare_fp_dataset(session: SpikeSession, sorter_names: List[str], metric_names: List[str], drop_nans: Optional[bool] = False):

    X = np.empty(shape=(0, len(metric_names)))
    y = np.array([])
    for sorter_name in sorter_names:
        metrics = session.get_sorter_metrics(sorter_name=sorter_name, metric_names=metric_names)
        ground_truth_comparison = session.get_ground_truth_comparison(sorter_name=sorter_name)

        fp_index = ground_truth_comparison.best_match_21 == -1

        X = np.vstack([X, metrics])
        y = np.append(y, fp_index)

    if drop_nans:
        nan_index = ~np.isnan(X).any(axis=1)
        X = X[nan_index]
        y = y[nan_index]

    return X, y

def _prepare_agreement_score_data(session: SpikeSession, sorter_names: List[str], hungarian: bool = True):
    #UNUSED
    y = np.array([])
    for sorter_name in sorter_names:
        ground_truth_comparison = session.get_ground_truth_comparison(sorter_name=sorter_name)
        if hungarian:
            agreement_scores = ground_truth_comparison.agreement_scores.iloc[ground_truth_comparison.best_match_21, :]
        else:
            agreement_scores = ground_truth_comparison.agreement_scores.max(axis=1)

        matched_agreement_scores = pd.Series(np.diag(agreement_scores))
        y = np.append(y, matched_agreement_scores)
    return y

# def prepare_dataset(
#         dataset_paths: List[Path],
#         sorter_names: List[str],
#         metric_names: List[str],
# ):
#     for dataset_path in dataset_paths:
#
#         data = [prepare_fp_dataset(session, sorter_names, metric_names) for dataset_path in dataset_paths]
#         return np.vstack([d[0] for d in data]), np.hstack([d[1] for d in data])

def prepare_dataset_from_hash(
        recording_path: str,
        gt_path: str,
        sorter_names: List[str],
        metric_names: List[str],
        cache_path: Path,
):
    cache_path = cache_path / recording_path.split('//')[1]

    recording = AutoRecordingExtractor(recording_path, download=True)
    gt_sorting = AutoSortingExtractor(gt_path)

    session = SpikeSession(recording, gt_sorting, cache_path=cache_path)
    return prepare_fp_dataset(session, sorter_names=sorter_names, metric_names=metric_names)