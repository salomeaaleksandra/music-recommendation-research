import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from textwrap import wrap
from collections import Counter
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, pairwise_distances
from imbalance_degree import imbalance_degree
from scipy.spatial.distance import euclidean, pdist, cdist

plt.style.use('seaborn-v0_8')


def write_file(path, list_to_write):
    with open(path, 'w') as fp:
        for item in list_to_write:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('file saved.')


def read_file(path):
    list_to_read = []
    with open(path, 'r') as fp:
        for line in fp:
            # remove linebreak from a current name
            # linebreak is the last character of each line
            x = line[:-1]

            # add current item to the list
            list_to_read.append(x)
    return list_to_read


def plot_elbow(k_max, features, plots_path=None):
    wcss = []
    for i in range(1, k_max + 1):
        model = KMeans(n_clusters=i, init="k-means++", n_init='auto', random_state=0)
        model.fit(features)
        wcss.append(model.inertia_)
        print('Model for {} clusters completed.'.format(i))
    plt.figure()
    plt.plot(range(1, k_max + 1), wcss, color='c')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    if plots_path is not None:
        plt.savefig(plots_path + r'\wcss_multiple_k.png', bbox_inches='tight', dpi=200)
    plt.show()


def calculate_cluster_centroids(data, y_pred):
    """
    Calculate cluster centroids based on predicted cluster labels.

    Parameters:
    data_points (numpy.ndarray): An array of shape (n_samples, n_features) containing data points.
    y_pred (numpy.ndarray): An array of shape (n_samples,) containing the predicted cluster labels for each data point.

    Returns:
    numpy.ndarray: An array of shape (n_clusters, n_features) containing the cluster centroids.
    """
    n_clusters = len(np.unique(y_pred))
    cluster_centroids = np.zeros((n_clusters, data.shape[1]))

    for cluster_label in range(n_clusters):
        cluster_mask = y_pred == cluster_label
        cluster_data = data[cluster_mask]
        cluster_centroid = np.mean(cluster_data, axis=0)
        cluster_centroids[cluster_label] = cluster_centroid

    return cluster_centroids


def calculate_distortion(data, cluster_centers):
    """
    Calculate the distortion of clustering using KMeans.cluster_centers_.

    Parameters:
    data_points (numpy.ndarray): An array of shape (n_samples, n_features) containing data points.
    cluster_centers (numpy.ndarray): An array of shape (n_clusters, n_features) containing cluster centers.

    Returns:
    float: The average of the squared distances from the cluster centers of the respective clusters to each data point.
    """
    valid_centers = cluster_centers[~np.isnan(cluster_centers).any(axis=1)]
    
    if valid_centers.shape[0] == 0:
        return np.nan  # No valid clusters
    
    # Compute squared Euclidean distances from each point to each valid cluster center
    distances = cdist(data, valid_centers, metric='sqeuclidean')
    
    # For each data point, find the squared distance to its nearest valid cluster center
    min_distances = np.min(distances, axis=1)
    
    # Overall distortion is the average of these minimal distances
    overall_distortion = np.mean(min_distances)
    return overall_distortion


def calculate_wcss(data, cluster_centers):
    """
    Calculate the Within-Cluster Sum of Squares (WCSS).

    Parameters:
    data (numpy.ndarray): Array of shape (n_samples, n_features) containing data points.
    cluster_centers (numpy.ndarray): Array of shape (n_clusters, n_features) containing cluster centers.

    Returns:
    float: The total WCSS.
    """
    # Filter out any cluster centers that contain NaN values
    valid_centers = cluster_centers[~np.isnan(cluster_centers).any(axis=1)]
    
    if valid_centers.shape[0] == 0:
        return np.nan  # No valid clusters
    
    # Compute squared Euclidean distances from each point to each valid cluster center
    distances = cdist(data, valid_centers, metric='sqeuclidean')
    
    # For each data point, find the squared distance to its nearest valid cluster center
    min_distances = np.min(distances, axis=1)
    
    # WCSS is the sum of these minimal squared distances
    wcss = np.sum(min_distances)
    return wcss

def dunn_index(data, y_pred):
    """
    Calculate the Dunn Index for clustering evaluation, ignoring empty clusters.

    Parameters:
    data (numpy.ndarray): Input data points of shape (n_samples, n_features).
    y_pred (numpy.ndarray): Predicted cluster assignments for each data point.

    Returns:
    float: Dunn Index value. If only one non-empty cluster exists, returns 0.0.
           If the maximum intra-cluster distance is 0 (e.g., all clusters have one point),
           returns np.inf.
    """
    # Get unique clusters from predictions (ignoring any labels that aren't present)
    unique_clusters = np.unique(y_pred)
    
    # If fewer than 2 clusters are present, Dunn index is undefined (or zero)
    if len(unique_clusters) < 2:
        return 0.0

    # Calculate centroids only for non-empty clusters
    centroids = []
    for cluster in unique_clusters:
        cluster_points = data[y_pred == cluster]
        if cluster_points.size > 0:
            centroids.append(np.mean(cluster_points, axis=0))
    centroids = np.array(centroids)
    
    # If we have less than 2 centroids, we cannot compute inter-cluster distances.
    if len(centroids) < 2:
        return 0.0

    # Compute the minimum inter-cluster distance (distance between centroids)
    inter_cluster_distance = np.min(pdist(centroids, metric='euclidean'))
    
    # Compute intra-cluster distances (diameter of each cluster)
    intra_cluster_dists = []
    for cluster in unique_clusters:
        cluster_points = data[y_pred == cluster]
        if len(cluster_points) > 1:
            # Maximum pairwise distance within the cluster
            intra_cluster_dists.append(np.max(pdist(cluster_points, metric='euclidean')))
        else:
            # For a single-point cluster, we define the diameter as 0.
            intra_cluster_dists.append(0)
    
    # Maximum intra-cluster distance over all clusters
    max_intra_cluster_distance = np.max(intra_cluster_dists)
    
    # If max intra-cluster distance is 0, then clusters are perfectly compact (or are singletons)
    # and the Dunn index is undefined (or can be considered infinite).
    if max_intra_cluster_distance == 0:
        return np.inf

    # Dunn Index is the ratio of the smallest inter-cluster distance to the largest intra-cluster diameter
    return inter_cluster_distance / max_intra_cluster_distance


def class_imbalance_metric(y_pred):
    """
    Calculate the class imbalance metric by computing the ratio between the number of classes
    with more instances than the average and the number of classes.

    Parameters:
    y_pred (list or numpy.ndarray): Predicted cluster assignments for each data point.

    Returns:
    float: The ratio of classes with less instances than the average. The value closer to 0 indicates severe class
    imbalance.
    """
    unique_classes, class_counts = np.unique(y_pred, return_counts=True)
    avg_instances = np.mean(class_counts)
    num_clusters = len(np.unique(y_pred))

    num_classes_above_avg = sum(count > avg_instances for count in class_counts)
    class_imbalance_metric = 1 - (num_classes_above_avg/num_clusters)
    return class_imbalance_metric


def class_imbalance_ratio(y_pred):
    """
    Calculate the class imbalance ratio as the ratio of the largest cluster size
    to the average cluster size in the predicted cluster assignments.

    Parameters:
    y_pred (list or numpy.ndarray): Predicted cluster assignments for each data point.

    Returns:
    float: Class imbalance ratio. The value closer to 1 indicate even distribution of clusters (where 1 is the minimum
    possible value)
    """

    unique_classes, class_counts = np.unique(y_pred, return_counts=True)
    max_count = np.max(class_counts)
    avg_count = np.mean(class_counts)

    imbalance_ratio = max_count / avg_count
    return imbalance_ratio


def imbalance_index(y_pred):
    """
    Calculate the imbalance index as the average absolute difference between
    the proportion of each class in the entire dataset and the proportion of each
    class within each cluster.

    Parameters:
    y_pred (list or numpy.ndarray): Predicted cluster assignments for each data point.

    Returns:
    float: Imbalance index. Value closer to 0 indicates a better class distribution.
    """
    prediction_counts = Counter(y_pred)
    num_instances = len(y_pred)
    num_clusters = len(np.unique(y_pred))

    class_proportions = np.array(list(prediction_counts.values())) / num_instances

    imbalance_sum = 0.0
    for cluster_id in range(num_clusters):
        cluster_size = prediction_counts[cluster_id]
        cluster_proportion = cluster_size / num_instances
        imbalance_sum += np.abs(cluster_proportion - class_proportions[cluster_id])

    imbalance_index = imbalance_sum / num_clusters
    return imbalance_index


def gini_coefficient(y_pred):
    """
    Compute the Gini coefficient to measure cluster imbalance.
    
    Parameters:
    y_pred (array-like): Cluster labels for each sample.
    
    Returns:
    float: Gini coefficient, where higher values indicate stronger imbalance.
    """
    unique, counts = np.unique(y_pred, return_counts=True)
    cluster_sizes = np.sort(counts)  # Sort cluster sizes for calculation
    n = len(cluster_sizes)
    
    if n <= 1:
        return 0  # If only one cluster exists, return 0 (fully balanced)
    
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * cluster_sizes)) / (n * np.sum(cluster_sizes)) - (n + 1) / n
    return gini
    

def evaluate_model(data, y_pred, imbalance_meausre='EU'):
    silhouette = silhouette_score(data, y_pred)
    print('Silhouette score: {}'.format(silhouette))
    ch = calinski_harabasz_score(data, y_pred)
    print('Calinski-Harabasz score: {}'.format(ch))
    dbi = davies_bouldin_score(data, y_pred)
    print('Davies-Bouldin score: {}'.format(dbi))
    di = dunn_index(data, y_pred)
    print('Dunn index: {}'.format(di))
    centroids = calculate_cluster_centroids(data, y_pred)
    dist = calculate_distortion(data, centroids)
    print('Distortion: {}'.format(dist))
    wcss = calculate_wcss(data, centroids)
    print('WCSS: {}'.format(wcss))
    imb = imbalance_degree(y_pred, imbalance_meausre)
    print('Class imbalance degree: {}'.format(imb))
    cim = class_imbalance_metric(y_pred)
    print('Class imbalance metric: {}'.format(cim))
    cir = class_imbalance_ratio(y_pred)
    print('Class imbalance ratio: {}'.format(cir))
    ii = imbalance_index(y_pred)
    print('Class imbalance index: {}'.format(ii))
    gini = gini_coefficient(y_pred)
    print('Gini index: {}'.format(gini))
    return silhouette, ch, dbi, di, dist, wcss, imb, cim, cir, ii, gini


def confusion_matrix_multilabel(y_pred, y_true):
    """ Compute a confusion matrix comparing genre labels with predicted clusters. """

    unique_labels = sorted(list(set(y_true)))  # Sort genres alphabetically
    clusters = sorted(list(set(y_pred)))  # Sort cluster IDs

    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    cluster_to_idx = {cluster: idx for idx, cluster in enumerate(clusters)}

    cm = np.zeros((len(unique_labels), len(clusters)), dtype=int)

    for i, true_label in enumerate(y_true):
        if true_label in label_to_idx and y_pred[i] in cluster_to_idx:
            row_idx = label_to_idx[true_label]
            col_idx = cluster_to_idx[y_pred[i]]
            cm[row_idx][col_idx] += 1

    return unique_labels, clusters, cm

def plot_cm(unique_labels, clusters, cm, normalize=True):
    fig, ax = plt.subplots(figsize=(10, 8))  # Increase figure size for better readability

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # Normalize by row sum
        cm = np.nan_to_num(cm)  # Replace NaNs that occur due to division by zero
    
    im = ax.imshow(cm, interpolation='nearest', cmap='crest')

    # Add colorbar
    ax.figure.colorbar(im, ax=ax)

    # Set ticks and labels
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=clusters,
        yticklabels=unique_labels,
        xlabel='Predicted Clusters',
        ylabel='True Genres',
    )

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=90, ha='right', rotation_mode='anchor')

    # Display values inside the matrix
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    
    ax.set_title('Normalized Confusion Matrix' if normalize else 'Confusion Matrix')
    plt.grid(False)
    plt.tight_layout()
    plt.show()



def plot_2D_PCA(X, y, title, plots_path=None, show_legend=True):
    X_PCA = PCA(n_components=2).fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Getting unique categories and assigning colors
    unique_categories = np.unique(y)
    num_categories = len(unique_categories)
    #colors = plt.cm.tab20b(np.linspace(0, 1, num_categories))
    colors = np.concatenate([plt.cm.tab20b(np.linspace(0, 1, num_categories // 2)),
                             plt.cm.tab20c(np.linspace(0, 1, num_categories - num_categories // 2))])

    # Create scatter plots for each category separately
    scatter_handles = []
    for i, category in enumerate(unique_categories):
        mask = (y == category)
        category_count = np.sum(mask)  # Calculate the count for the current category
        scatter = ax.scatter(X_PCA[mask, 0], X_PCA[mask, 1], c=[colors[i]],
                             label=f'{category} ({category_count})', s=2)
        scatter_handles.append(scatter)

    # Combine legend handles and labels for all categories
    legend_labels = [handle.get_label() for handle in scatter_handles]

    # Display legend
    if show_legend:
        legend = ax.legend(handles=scatter_handles, labels=legend_labels, markerscale=5, title="Class (count)",
                  loc='upper left', bbox_to_anchor=(1, 1.05))

    plt.grid(True)
    plt.xlabel('PCA index 0')
    plt.ylabel('PCA index 1')
    plt.title(title)
    plt.tight_layout()
    if plots_path is not None:
        if show_legend:
            plt.savefig(r'{}\PCA2D_{}.png'.format(plots_path, title),
                        bbox_extra_artists=(legend,),
                        bbox_inches='tight', dpi=200)
        else:
            plt.savefig(r'{}\PCA2D_{}.png'.format(plots_path, title),
                            bbox_inches='tight', dpi=200)            
    #plt.show()

def plot_3D_PCA(X, y, title, plots_path=None):
    """
    Plots a 3D PCA representation of the dataset.

    Args:
    X (array-like): The input data matrix.
    y (array-like): Target labels.
    title (str): Title of the plot.
    plots_path (str, optional): Path to save the plot. If None, the plot will not be saved.
    """
    X_PCA = PCA(n_components=3).fit_transform(X)  # Compute 3D PCA
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Getting unique categories and assigning colors
    unique_categories = np.unique(y)
    num_categories = len(unique_categories)
    colors = np.concatenate([plt.cm.tab20b(np.linspace(0, 1, num_categories // 2)),
                             plt.cm.tab20c(np.linspace(0, 1, num_categories - num_categories // 2))])

    # Create scatter plots for each category separately
    scatter_handles = []
    for i, category in enumerate(unique_categories):
        mask = (y == category)
        category_count = np.sum(mask)  # Calculate the count for the current category
        scatter = ax.scatter(X_PCA[mask, 0], X_PCA[mask, 1], X_PCA[mask, 2], c=[colors[i]],
                              label=f'{category} ({category_count})', s=2, alpha=0.5, depthshade=True)
        scatter_handles.append(scatter)

    # Combine legend handles and labels for all categories
    legend_labels = [handle.get_label() for handle in scatter_handles]

    # Display legend
    legend = ax.legend(handles=scatter_handles, labels=legend_labels, markerscale=2, title="Class (count)",
                       loc='upper left', bbox_to_anchor=(1.05, 1))

    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    ax.set_title(title)
    plt.tight_layout()

    # Save the plot if a path is provided
    if plots_path is not None:
        plt.savefig(r'{}\PCA3D_{}.png'.format(plots_path, title),
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight', dpi=200)
    plt.show()


def plot_bar_score(scores, score_name, model_names, plots_path=None):
    x_axis = np.arange(len(model_names))
    ax = plt.subplot()
    plt.bar(x_axis, scores, color='c')
    ax.set_xticks(x_axis)
    ax.set_xticklabels(model_names, fontsize=6, rotation=75)
    plt.grid()
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.title("{} score for clustering".format(score_name))
    if plots_path is not None:
        plt.savefig(r'{}\{}.png'.format(plots_path, score_name), bbox_inches='tight', dpi=200)
    plt.show()


def plot_cm_scores(x, y, cm, x_axis, y_axis, title, plots_path=None):
    fig, ax = plt.subplots(figsize=(6, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap='viridis') #plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    x_labels = ['\n'.join(wrap(l, 25)) for l in x]
    # set ticks and labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=x_labels,
           yticklabels=y,
           xlabel=x_axis,
           ylabel=y_axis)
    # rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=90, ha='right', va='center', rotation_mode='anchor', fontsize=5)

    # values in plot
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, s="%.2f" % cm[i, j], ha='center', va='center', color='w', fontsize=6)

    ax.set_title(title)
    plt.grid(False)
    if plots_path is not None:
        plt.savefig(r'{}\{}.png'.format(plots_path, title), bbox_inches='tight', dpi=200)
    plt.tight_layout()
    plt.show()


def assign_column_place(dataframe, sort_order_list):
    """
    Assigns a ranking (place) to each column in a DataFrame based on sorting order.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame to process.
    sort_order_list (list): A list specifying sort order for each column ('asc' for ascending, 'desc' for descending).

    Returns:
    pd.DataFrame: A DataFrame where values are replaced by their ranking positions.
    """
    if not isinstance(dataframe, pd.DataFrame) or len(dataframe.columns) != len(sort_order_list):
        raise ValueError("Input should be a Pandas DataFrame with the same number of columns as elements in sort_order_list.")

    valid_sort_orders = ["asc", "ascending", "desc", "descending"]
    for order in sort_order_list:
        if order.lower() not in valid_sort_orders:
            raise ValueError("Invalid sort order. Use 'asc', 'ascending', 'desc', or 'descending'.")

    # Create a copy to avoid modifying the original DataFrame
    ranked_dataframe = dataframe.copy()

    # Rank each column based on sorting order
    for column, order in zip(ranked_dataframe.columns, sort_order_list):
        ascending = order.lower() in ["asc", "ascending"]
        ranked_dataframe[column] = ranked_dataframe[column].rank(method='min', ascending=ascending)

    return ranked_dataframe


