# clustering of signal derived from time series telemetry data.

from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesResampler

import matplotlib.pyplot as plt

# downsample to get same lengths signal to put into an array that can be used for clustering.
# ensure X_train is reshaped into nx1 array for clustering with tslearn
def	k_means_clustering(X_train, num_clusters = 15, metric = "euclidean"): # check other metrics like dtw
	seed = 0
	X_train = TimeSeriesResampler(sz=350).fit_transform(X_train)
	km = TimeSeriesKMeans(n_clusters=num_clusters, metric=metric, verbose=True, random_state=seed)
	y_pred = km.fit_predict(X_train)
	return km, y_pred

def plot_clusters(X_train, km, y_pred):
    plt.figure(figsize=(10,6))
    for yi in range(15):
        plt.subplot(3, 5, yi + 1)
        for xx in X_train[y_pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.2)
        plt.plot(km.cluster_centers_[yi].ravel(), "r-")
        plt.text(0.55, 0.85,'C %d' % (yi),
                transform=plt.gca().transAxes)
        if yi == 1:
            plt.title("Euclidean $k$-means")
    return plt