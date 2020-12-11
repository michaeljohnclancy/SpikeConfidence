from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from sorter import SpikeSorter


def prepare_hungarian_agreement_dataset(
        dataset_path: Path,
        sorter_names: List[str],
        metric_names: List[str],
        drop_nans: bool = True
):
    session = SpikeSorter(dataset_path)

    X = np.empty(shape=(0, len(metric_names)))
    y = np.array([])
    for sorter_name in sorter_names:
        metrics = session.get_sorter_metrics(sorter_name=sorter_name, metric_names=metric_names)
        ground_truth_comparison = session.get_ground_truth_comparison(sorter_name=sorter_name)

        agreement_scores = ground_truth_comparison.agreement_scores.iloc[ground_truth_comparison.best_match_21, :]
        hungarian_agreement_scores = pd.Series(np.diag(agreement_scores))

        X = np.vstack([X, metrics])
        y = np.append(y, hungarian_agreement_scores)

    if drop_nans:
        nan_index = ~np.isnan(X).any(axis=1)
        X = X[nan_index]
        y = y[nan_index]

    return X, y
