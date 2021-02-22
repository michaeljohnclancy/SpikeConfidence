import os
import numpy as np
from pathlib import Path
import spikesorters as ss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import kachery as ka

from utils import prepare_dataset_from_hash

#For the k most important features, plot the training accuracies.
#As the input data is standardised, the coefficients can be ranked by their absolute value.

#We can retrain the logistic regression with the reduced metric set,
#and as a result can assess the redundancy of the metrics.


# Default TMP directory may have reach storage capacity whilst running spike sorters. Specify your own tmp dir here.
tmp_path = Path('/home/mclancy/SpikeConfidence/.tmp/')
os.environ["TMP"] = str(tmp_path)
os.environ["TMPDIR"] = str(tmp_path)
os.environ["TEMPDIR"] = str(tmp_path)
os.environ["ML_TEMPORARY_DIRECTORY"] = str(tmp_path)
tmp_path.mkdir(exist_ok=True, parents=True)

# This is the cache currently holding the sortings and metrics for recordings.
cache_path = Path('/data/.cache')
cache_path.mkdir(exist_ok=True, parents=True)

# Need to create this folder before use.
kache_path = Path("/data/.kache")
kache_path.mkdir(exist_ok=True, parents=True)
os.environ["KACHERY_STORAGE_DIR"] = str(kache_path)

# Configure kachery to download data from the public database
ka.set_config(fr='default_readonly')

base_dir = Path('/home/mclancy/SpikeConfidence/')
spike_sorter_dir = base_dir / 'spikesorters'

# Specify the path to the non python sorters.
ss.Kilosort2_5Sorter.set_kilosort2_5_path(spike_sorter_dir / 'Kilosort')
ss.IronClustSorter.set_ironclust_path(spike_sorter_dir / 'ironclust')

# kilosort2_5
ss.Kilosort2_5Sorter.set_kilosort2_5_path(spike_sorter_dir / 'Kilosort')

ss.IronClustSorter.set_ironclust_path(spike_sorter_dir / 'ironclust')

drift_recording_path = 'sha1dir://615aa23efde8898aa89002613e20ad59dcde42f9.hybrid_janelia/drift_siprobe/rec_16c_1200s_11'
drift_gt_path = 'sha1dir://615aa23efde8898aa89002613e20ad59dcde42f9.hybrid_janelia/drift_siprobe/rec_16c_1200s_11/firings_true.mda'

sorter_names = ['mountainsort4', 'herdingspikes', 'spykingcircus', 'ironclust','tridesclous']

metric_names = np.array(["num_spikes", "firing_rate", "presence_ratio",
                         "isi_violation", "amplitude_cutoff", "snr",
                         "max_drift", "cumulative_drift", "silhouette_score",
                         "isolation_distance", "l_ratio",
                         "nn_hit_rate", "nn_miss_rate","d_prime"])

# False positive classification dataset prepared
X, y = prepare_dataset_from_hash(recording_path=drift_recording_path, gt_path=drift_gt_path, metric_names=metric_names, sorter_names=sorter_names, cache_path=cache_path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(X_train, y_train)

ranked_metrics_index = sorted(range(len(metric_names)), key=lambda k: abs(model.named_steps.logisticregression.coef_[0][k]), reverse=True)

# Plotting the regression accuracies for the k most important features, for k from 1 to num_metrics,
accuracies = []
for i in range(1, len(ranked_metrics_index)+1):
    m_names = metric_names[ranked_metrics_index[:i]]
    print(f"Features are {m_names}")

    X, y = prepare_dataset_from_hash(recording_path=drift_recording_path, gt_path=drift_gt_path,
                                     metric_names=m_names, sorter_names=sorter_names, cache_path=cache_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    # False positive classification via logistic regression
    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    accuracies.append(acc)
    # model_f1_score = f1_score(y_test, y_preds, average='binary')
    print(f"Acc Score is {acc}")


plt.clf()
plt.plot(range(len(metric_names)), accuracies)
plt.suptitle("Logistic regression performances with best k features")
plt.xlabel('k (num features)')
plt.ylabel('Accuracy')
plt.savefig('figures/featureselection.pdf')

# plt.xticks(np.arange(1, len(metric_names)+1))
# plt.yticks(np.arange(0.84, 0.95, 0.01))
