import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


def dbscan(epsilon, minpoints, file_location):
    # Load the data
    data = pd.read_csv(file_location)

    # Assuming the data has only the features for clustering
    X = np.array(data)

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=epsilon, min_samples=minpoints)
    dbscan.fit(X)

    # Extract labels and core samples
    labels = dbscan.labels_
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True

    # Number of clusters in labels, ignoring noise if present
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print(f'Estimated number of clusters: {n_clusters_}')
    print(f'Estimated number of noise points: {n_noise_}')

    # Add the labels to the dataframe
    data['Cluster'] = labels

    # Save the clustered data to a new CSV file
    output_file = file_location.replace('.csv', '_clustered.csv')
    data.to_csv(output_file, index=False)

    # Plot the results
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title(f'Estimated number of clusters: {n_clusters_}')
    plt.show()
