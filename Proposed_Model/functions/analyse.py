import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

def calculate_spear(array_1, array_2):

    cpu_array_1 = array_1.clone().detach().cpu().numpy()
    cpu_array_2 = array_2.clone().detach().cpu().numpy()
    correlation, _ = spearmanr(cpu_array_1, cpu_array_2)

    return correlation

def calculate_euclidean(array_1, array_2):

    cpu_array_1 = array_1.clone().detach().cpu().numpy()
    cpu_array_2 = array_2.clone().detach().cpu().numpy()

    return np.linalg.norm(cpu_array_1 - cpu_array_2)

def calculate_pca(mu, noise, epoch):

    latent_mus = mu.clone().detach().cpu().numpy()
    latent_noises = noise.rsample().clone().detach().cpu().numpy()
    pca = PCA(n_components=2)
    mu_2d = pca.fit_transform(latent_mus)
    noise_2d = pca.fit_transform(latent_noises)

    plt.figure(figsize=(8, 6))
    plt.scatter(mu_2d[:, 0], mu_2d[:, 1], c='blue', alpha=0.6, label='mu', marker='o')
    plt.scatter(noise_2d[:, 0], noise_2d[:, 1], c='red', alpha=0.4, label='noise (sample)', marker='x')
    plt.title('PCA of Latent Space')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'pca_visualization_{epoch}.png', dpi=300)

def bin_data_to_tags(tags, bin_num):
    nums = [round(float(string.split('_')[-1]), 3) for string in tags]
    _, new_tag = np.histogram(nums, bin_num)
    indices = np.digitize(nums, new_tag)
    #print(bin_num, nums, new_tag)
    result = [f'>={round(new_tag[index-1], 3)}' for index in indices]

    return result

def show_curve(arrs, labels, x_label='Dimension Index', y_label='Variance (Normalized)'):

    plt.figure(figsize=(10, 6))

    for i in range(len(arrs)):
        plt.plot(arrs[i], label=labels[i]) 

    plt.legend()
    plt.title('')
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.show()

def show_curves(indices_try_list, num_try_list, tensions, distances, strains):

    fig, axes = plt.subplots(len(indices_try_list), len(num_try_list), figsize=(14, 14))
    axes = axes.ravel()

    for i in range(len(indices_try_list)):
        for j in range(len(num_try_list)):
            ax = axes[i*len(indices_try_list)+j]

            ax.plot(tensions[i][j], 'r-', label='Tension') 
            ax.plot(distances[i][j], 'g-', label='Distance') 
            ax.plot(strains[i][j], 'b-', label='Strain') 

            ax.set_ylim(0, 4)

            ax.set_title(num_try_list[j])
            ax.grid(True)

    #plt.legend()
    #plt.title('Three Arrays Visualization')
    #plt.xlabel('Index')
    #plt.ylabel('Value')
    plt.tight_layout()

    plt.show()

def major_component(z):
    z = z.clone().detach().cpu().numpy()

    '''
    pca = PCA()
    pca.fit_transform(z)

    '''
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(z)

    pca = PCA()
    pca.fit(X_scaled)
    

    components = pca.components_ 
    explained_variance = pca.explained_variance_  
    explained_variance_ratio = pca.explained_variance_ratio_ 

    pca_results = pd.DataFrame({
        'Component': range(1, len(explained_variance_ratio)+1),
        'Eigenvalue': explained_variance,
        'Variance Explained': explained_variance_ratio,
        'Cumulative Variance': np.cumsum(explained_variance_ratio)
    })

    pca_results_sorted = pca_results.sort_values('Eigenvalue', ascending=False)

    print("所有主成分按解释方差排序:")
    print(pca_results_sorted)

    return explained_variance

#show_curve()