import json
import shutil
from pathlib import Path
from typing import List, Dict
import spikeextractors as se
from spikecomparison.studytools import load_probe_file_inplace
from spikesorters import run_sorter
from spikeextractors import MEArecRecordingExtractor, NpzSortingExtractor, MEArecSortingExtractor, SortingExtractor


class SpikeSorter:
    def __init__(self, dataset_path: Path,
                 sorter_names: List[str]):

        self.dataset_path = dataset_path
        self.sorter_names = sorter_names
        self.cache_path = dataset_path.parent / '.cache'
        self.recording_cache = self.cache_path / 'recording'
        self.ground_truth_cache = self.cache_path / 'ground_truth.npz'
        self.sorting_cache = self.cache_path / 'sortings'
        self.metrics_cache = self.cache_path / 'metrics.feather'
        self.raw_data_path = self.recording_cache / 'raw.dat'
        self.prb_path = self.recording_cache / 'probes.prb'
        self.info_path = self.recording_cache / 'info.json'

        self.recording = self.get_recording()
        self.ground_truth = self.get_ground_truth()

    def get_recording(self) -> MEArecRecordingExtractor:
        try:
            self.recording_cache.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            try:
                recording_extractor = self._get_cached_recording()
            except:
                print("Error reading recording, deleting")
                # If the recording cache exists but cant be read,
                # the whole cache including the sorting results are invalidated.
                shutil.rmtree(self.cache_path)
                return self.get_recording()
        else:
            recording_extractor = MEArecRecordingExtractor(self.dataset_path)
            self._cache_recording(recording_extractor)

        return recording_extractor

    def get_ground_truth(self) -> NpzSortingExtractor:
        if self.ground_truth_cache.exists():
            try:
                return NpzSortingExtractor(self.ground_truth_cache)
            except :
                print(f"Error reading ground truth cache, deleting...")
                self.ground_truth_cache.unlink()
                return self.get_ground_truth()
        else:
            ground_truth = MEArecSortingExtractor(self.dataset_path)
            NpzSortingExtractor.write_sorting(ground_truth, save_path=str(self.ground_truth_cache))
            return NpzSortingExtractor(self.ground_truth_cache)

    def get_sortings(self) -> Dict[str, NpzSortingExtractor]:
        return {sorter_name: self.get_sorting(sorter_name) for sorter_name in self.sorter_names}

    def get_sorting(self, sorter_name) -> NpzSortingExtractor:
        s_cache = self.sorting_cache / f'{sorter_name}.npz'
        if s_cache.exists():
            try:
                return NpzSortingExtractor(s_cache)
            except:
                print(f"Error reading {sorter_name} cache, deleting...")
                shutil.rmtree(s_cache)
                return self.get_sorting(sorter_name)

        else:
            print(f"Running {sorter_name}...")
            sorting: SortingExtractor = run_sorter(
                sorter_name,
                recording=self.recording,
                output_folder=s_cache,
                delete_output_folder=True
            )

            NpzSortingExtractor.write_sorting(sorting, save_path=str(s_cache))
            return NpzSortingExtractor(s_cache)

    # def get_sorter_metrics(self, sorter_name: str, metrics_names: List[str]):
    #     m_cache = self.sorting_cache / sorter_name / 'metrics'
    #     try:
    #         m_cache.mkdir(parents=True, exist_ok=False)
    #     except FileExistsError:
    #         sorter_class = sorter_dict[sorter_name]

    def _cache_recording(self, recording_extractor):
        """ If the recording exists in the cache, just return it.
        Otherwise, create it from the dataset, cache it, and return it.
        """
        # Caches info
        sample_rate = recording_extractor.get_sampling_frequency()
        num_chan = recording_extractor.get_num_channels()
        info = dict(sample_rate=sample_rate, num_chan=num_chan, dtype='float32', time_axis=0)
        with open(self.info_path, 'w', encoding='utf8') as f:
            json.dump(info, f, indent=4)

        # Caches raw data
        chunk_size = 2 ** 24 // num_chan
        recording_extractor.write_to_binary_dat_format(
            str(self.raw_data_path),
            time_axis=0,
            dtype='float32',
            chunk_size=chunk_size
        )
        # Caches prb
        recording_extractor.save_to_probe_file(str(self.prb_path))

    def _get_cached_recording(self):
        with open(self.info_path, 'r', encoding='utf8') as f:
            info = json.load(f)
        rec = se.BinDatRecordingExtractor(self.raw_data_path, info['sample_rate'], info['num_chan'],
                                          info['dtype'], time_axis=info['time_axis'])
        load_probe_file_inplace(rec, str(self.prb_path))
        return rec
