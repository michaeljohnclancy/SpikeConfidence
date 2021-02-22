import pandas as pd
import json
import shutil
from pathlib import Path
from typing import List, Dict
import spiketoolkit as st
from spikecomparison import GroundTruthComparison
from spikesorters import run_sorter
from spikeextractors import NpzSortingExtractor, RecordingExtractor, SortingExtractor


class SpikeSession:

    def __init__(
            self,
            recording_extractor: RecordingExtractor,
            gt_sorting_extractor: SortingExtractor,
            cache_path: Path
    ):

        cache_path.mkdir(parents=True, exist_ok=True)

        self.sorting_cache = cache_path / 'sortings'
        self.metrics_cache = cache_path / 'metrics'

        self.recording = recording_extractor
        self.ground_truth = gt_sorting_extractor

    def get_sortings(self, sorter_names: List[str]) -> Dict[str, NpzSortingExtractor]:
        return {sorter_name: self.get_sorting(sorter_name) for sorter_name in sorter_names}

    def get_sorting(self, sorter_name) -> NpzSortingExtractor:
        s_cache = self.sorting_cache / sorter_name
        try:
            shutil.rmtree(s_cache)
        except FileNotFoundError:
            pass
        s_npz_cache = s_cache.with_suffix('.npz')
        if s_npz_cache.exists():
            try:
                return NpzSortingExtractor(s_npz_cache)
            except:
                print(f"Error reading {sorter_name} cache, deleting...")
                s_npz_cache.unlink()
                return self.get_sorting(sorter_name)

        else:
            print(f"Running {sorter_name}...")
            s_cache.mkdir(parents=True)
            sorting = run_sorter(
                sorter_name,
                recording=self.recording,
                output_folder=s_cache,
            )

            NpzSortingExtractor.write_sorting(sorting, save_path=str(s_npz_cache))

            shutil.rmtree(s_cache)

            return NpzSortingExtractor(s_npz_cache)

    def get_sorter_metrics(self, sorter_name: str, metric_names: List[str]):
        self.metrics_cache.mkdir(exist_ok=True, parents=True)
        metrics_cache = self.metrics_cache / f'{sorter_name}.parquet'
        sorting = self.get_sorting(sorter_name)
        if metrics_cache.exists():
            metrics = pd.read_parquet(metrics_cache)
            metrics_to_calc = [m for m in metric_names if m not in metrics.columns]

            if metrics_to_calc:
                new_metrics = st.validation.compute_quality_metrics(
                    sorting, self.recording,
                    metric_names=metrics_to_calc,
                    as_dataframe=True
                )
                metrics = pd.concat([metrics, new_metrics], axis=1)
                metrics_cache.unlink()
                metrics.to_parquet(str(metrics_cache))

            return metrics[metric_names]
        else:
            metrics = st.validation.compute_quality_metrics(
                sorting, self.recording,
                metric_names=metric_names,
                as_dataframe=True)
            metrics.to_parquet(str(metrics_cache))
            return metrics

    def get_ground_truth_comparison(self, sorter_name: str) -> GroundTruthComparison:
        sorting = self.get_sorting(sorter_name)
        return GroundTruthComparison(self.ground_truth, sorting)
