import json
import shutil
from pathlib import Path
from typing import List, Type
import pandas as pd
import spiketoolkit as st
import spikeextractors as se
from spikesorters import sorter_dict, run_sorter
from spikecomparison.studytools import load_probe_file_inplace
from spikeextractors import BinDatRecordingExtractor, MEArecRecordingExtractor, NpzSortingExtractor
class SpikeSorter:
    def __init__(self, dataset_path: Path, cache_path: Path, sorter_names: List[str], 
                 recording_extractor_class: Type[BinDatRecordingExtractor] = MEArecRecordingExtractor):
        self.dataset_path = dataset_path
        self.cache_path = cache_path
        self.sorter_names = sorter_names
        self.recording_extractor_class = recording_extractor_class
        cache_path = cache_path / f"{dataset_path.name.split('.', -1)[0]}"
        self.recording_cache = cache_path / 'recording'
        self.sorting_cache = cache_path / 'sortings'
        self.raw_data_path = self.recording_cache / 'raw.dat'
        self.prb_path = self.recording_cache / 'probes.prb'
        self.info_path = self.recording_cache / 'info.json'
        self._build_recording()
        self._build_sorting_extractors()
    def _build_recording(self):
        try:
            self.recording_cache.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            try:
               self.recording_extractor = self._get_cached_recording()
            except:
                print("Error reading cache, deleting")
                shutil.rmtree(self.recording_cache)
                self._build_recording()
        else:
            self.recording_extractor = self.recording_extractor_class(self.dataset_path)
            self._cache_recording()
    def _build_sorting_extractors(self):
            sorting_extractors = {}
            for sorter_name in self.sorter_names:
                s_cache = self.sorting_cache / sorter_name
                try:
                    s_cache.mkdir(parents=True, exist_ok=False)
                except FileExistsError:
                    sorter_class = sorter_dict[sorter_name]
                    sorting_extractors[sorter_name] = sorter_class.get_result_from_folder(s_cache)
                else:
                    sorting_extractors[sorter_name] = run_sorter(sorter_name, recording=self.recording_extractor, output_folder=s_cache)
            self.sorting_extractors = sorting_extractors
    def get_recording(self):
        return self.recording_extractor
    def get_sorting(self, sorter_name: str) -> NpzSortingExtractor:
            return self.sorting_extractors[sorter_name]
    def _cache_recording(self):
        """ If the recording exists in the cache, just return it.
        Otherwise, create it from the dataset, cache it, and return it.
        """
        # Caches info
        sample_rate = self.recording_extractor.get_sampling_frequency()
        num_chan = self.recording_extractor.get_num_channels()
        info = dict(sample_rate=sample_rate, num_chan=num_chan, dtype='float32', time_axis=0)
        with open(self.info_path, 'w', encoding='utf8') as f:
            json.dump(info, f, indent=4)
        # Caches raw data
        chunk_size = 2 ** 24 // num_chan
        self.recording_extractor.write_to_binary_dat_format(str(self.raw_data_path), time_axis=0, dtype='float32', chunk_size=chunk_size)
        # Caches prb
        self.recording_extractor.save_to_probe_file(str(self.prb_path))
    def _get_cached_recording(self):
        with open(self.info_path, 'r', encoding='utf8') as f:
            info = json.load(f)
        rec = se.BinDatRecordingExtractor(self.raw_data_path, info['sample_rate'], info['num_chan'],
                                          info['dtype'], time_axis=info['time_axis'])
        load_probe_file_inplace(rec, str(self.prb_path))
        return rec
    def _get_cached_sorting(self, cache_path: Path):
        return se.NpzSortingExtractor(cache_path)

def example():
    cache_path = Path("/home/mclancy/SpikeConfidence/analyses/recordings_50cells_SqMEA_2020/cachingtest")
    dataset_path = Path("/home/mclancy/SpikeConfidence/analyses/recordings_50cells_SqMEA_2020/recordings_50cells_SqMEA-10-15_600.0_10.0uV_21-01-2020_18-12.h5")
    session = SpikeSorter(dataset_path, cache_path, sorter_names=['herdingspikes'])
    recording = session.get_recording()
    sorting = session.get_sorting(sorter_name='herdingspikes')
    print(sorting)
    print(recording)
example()
