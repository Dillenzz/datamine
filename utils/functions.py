import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgba
from sklearn.decomposition import PCA

def evaluate_clusters(metric, data, labels):
    """ Evaluates the clustering results
    
    ---------------
    Note that noise points have to be discarded, can be done via e.g. for model
    model.labels_
    
    mask = model.labels_ != -1
    data = clustering_data[mask]
    labels = model.labels_[mask]
    ---------------
    
    Params:
    metric: 
        a function for measuring cluster quality. E.g. davies_bouldin_score
        
    data: 
        dataframe or numpy array of data
        
    labels: 
        assigned labels by clustering algorightm """
    if len(np.unique(labels)) > 1:
        score = metric(data, labels)
        print(f'{metric.__name__}: {score}')
        return score
    else:
        print('Score cannot be calculated, only one cluster found.')
        
def plot_clusters_2d(model, clustering_data, full_data, noise=False):
    """ Plots the clustering in 2D PCA with improved colormap """
    full_data['Cluster'] = model.labels_
    
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(clustering_data)
    
    representative_countries = full_data.groupby('Cluster')['Location'].first()
    
    unique_clusters = np.unique(model.labels_)
    num_clusters = len(unique_clusters) - (1 if noise else 0)
    
    # Generate colors using viridis, but avoid white and very light colors
    viridis = plt.cm.get_cmap('viridis')
    cluster_colors = viridis(np.linspace(0.1, 0.9, num_clusters))
    
    if noise:
        # Add gray for noise, ensuring it's the first color
        colors = np.vstack(([0.5, 0.5, 0.5, 1], cluster_colors))
        discrete_cmap = ListedColormap(colors)
        cluster_colors = np.where(model.labels_ == -1, 0, model.labels_ + 1)
    else:
        discrete_cmap = ListedColormap(cluster_colors)
        cluster_colors = model.labels_
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_components[:, 0], pca_components[:, 1],
                          c=cluster_colors, cmap=discrete_cmap, s=50,
                          alpha=0.7)
    
    for cluster, country in representative_countries.items():
        if cluster != -1 or not noise:
            idx = full_data[full_data['Location'] == country].index[0]
            x, y = pca_components[idx, 0], pca_components[idx, 1]
            plt.annotate(country, (x, y),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=10, weight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7),
            )
    
    plt.title(f'{type(model).__name__} clustering Projected with 2D PCA', fontsize=16)
    plt.xlabel('PCA Component 1', fontsize=12)
    plt.ylabel('PCA Component 2', fontsize=12)
    
    colorbar = plt.colorbar(scatter, aspect=30)
    if noise:
        colorbar.set_ticks(np.arange(len(unique_clusters)))
        colorbar.set_ticklabels(['Noise' if cluster == -1 else f'Cluster {cluster} ({country})' for cluster, country in representative_countries.items()])
    else:
        colorbar.set_ticks(np.arange(num_clusters))
        colorbar.set_ticklabels([f'Cluster {cluster} ({country})' for cluster, country in representative_countries.items()])

    
    plt.tight_layout()
    plt.show()