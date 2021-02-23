
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import kachery as ka
import os
from utils import prepare_dataset_from_hash
import numpy as np
import spikesorters as ss

# Default TMP directory may have reach storage capacity whilst running spike sorters. Specify your own tmp dir here.
storage_path = Path('/disk/scratch/mclancy')

tmp_path = storage_path / '.tmp'
os.environ["TMP"] = str(tmp_path)
os.environ["TMPDIR"] = str(tmp_path)
os.environ["TEMPDIR"] = str(tmp_path)
os.environ["ML_TEMPORARY_DIRECTORY"] = str(tmp_path)
tmp_path.mkdir(exist_ok=True, parents=True)

cache_path = storage_path / '.cache'

# Need to create this folder before use.
kache_path = storage_path / '.kache'
os.environ["KACHERY_STORAGE_DIR"] = str(kache_path)
kache_path.mkdir(exist_ok=True, parents=True)

# Configure kachery to download data from the public database
ka.set_config(fr='default_readonly')

spike_sorter_dir = Path('/disk/scratch/mhennig/spikeinterface')

# Specify the path to the non python sorters.
ss.KilosortSorter.set_kilosort_path(spike_sorter_dir / 'KiloSort')
ss.Kilosort2Sorter.set_kilosort2_path(spike_sorter_dir / 'Kilosort2')
ss.HDSortSorter.set_hdsort_path(spike_sorter_dir / 'HDsort')
ss.IronClustSorter.set_ironclust_path(spike_sorter_dir / 'ironclust')


sorter_names = ['kilosort', 'kilosort2', 'mountainsort4', 'herdingspikes', 'spykingcircus', 'ironclust', 'tridesclous', 'klusta', 'waveclus', 'combinato']

metric_names = np.array(["num_spikes", "firing_rate", "presence_ratio",
                         "isi_violation", "amplitude_cutoff", "snr",
                         "max_drift", "cumulative_drift", "silhouette_score",
                         "isolation_distance", "l_ratio",
                         "nn_hit_rate", "nn_miss_rate","d_prime"])


static_siprobe1_recording_path = 'sha1dir://615aa23efde8898aa89002613e20ad59dcde42f9.hybrid_janelia/static_siprobe/rec_16c_1200s_11'
static_siprobe1_gt_path = 'sha1dir://615aa23efde8898aa89002613e20ad59dcde42f9.hybrid_janelia/static_siprobe/rec_16c_1200s_11/firings_true.mda'

static_siprobe2_recording_path = 'sha1dir://615aa23efde8898aa89002613e20ad59dcde42f9.hybrid_janelia/static_siprobe/rec_16c_1200s_21'
static_siprobe2_gt_path = 'sha1dir://615aa23efde8898aa89002613e20ad59dcde42f9.hybrid_janelia/static_siprobe/rec_16c_1200s_21/firings_true.mda'

static_siprobe3_recording_path = 'sha1dir://615aa23efde8898aa89002613e20ad59dcde42f9.hybrid_janelia/static_siprobe/rec_16c_1200s_31'
static_siprobe3_gt_path = 'sha1dir://615aa23efde8898aa89002613e20ad59dcde42f9.hybrid_janelia/static_siprobe/rec_16c_1200s_31/firings_true.mda'

recording_paths = [static_siprobe1_recording_path, static_siprobe2_recording_path, static_siprobe3_recording_path]
gt_paths = [static_siprobe1_gt_path, static_siprobe2_gt_path, static_siprobe3_gt_path]

accuracies = {}
for sorter_name in sorter_names:
    X, y = prepare_dataset_from_hash(recording_paths=recording_paths, gt_paths=gt_paths, metric_names=metric_names, sorter_names=[sorter_name], cache_path=cache_path)

    # Shuffled and split into train/test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    # False positive classification via logistic regression
    model = make_pipeline(StandardScaler(), LogisticRegression())

    try:
        model.fit(X_train, y_train)
        model_accuracy = model.score(X_test, y_test)
        accuracies[sorter_name] = model_accuracy
    except ValueError as e:
        accuracies[sorter_name] = None
        print(f"Could not build regressor for {sorter_name} with data provided; Likely that there are no instances of false positives from this sorter for these data", e)

np.save('results/single_sorter_accuracies.npy', accuracies)