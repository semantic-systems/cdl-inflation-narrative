import os 
import polars as pl
from pathlib import Path
from elfen.extractor import Extractor
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


os.environ['POLARS_MAX_THREADS'] = '8'


def extract_features(input_csv_path, output_csv_path, normalized=True):
    df = pl.read_csv(input_csv_path)
    csv_columns = df.columns
    extractor = Extractor(data = df)
    feature_areas = ["surface", "pos", "lexical_richness", "readability", "information", "entities", "semantic", "emotion", "psycholinguistic", "morphological", "dependency"]
    extractor.extract_feature_group(feature_group = feature_areas)
    if normalized:
        extractor.normalize("all")
    feature_columns = [x for x in extractor.data.columns if x not in csv_columns]
    extractor.write_csv(output_csv_path)
    return feature_columns


def read_features(input_csv_path, feature_columns, l2_normalize=False):
    df = pd.read_csv(input_csv_path)
    feature = df[feature_columns].to_numpy()
    print(f"Read features with shape: {feature.shape}")
    feature[np.isnan(feature)] = 0
    feature[np.isinf(feature)] = 0
    if l2_normalize:
        feature = feature / np.linalg.norm(feature, axis=1, keepdims=True)
        print("shape after normalization:", feature.shape)
    return feature

def run_tsne(data, metric='cosine'):
    tsne = TSNE(n_components=2, random_state=42, perplexity=5, max_iter=1000, metric=metric)
    tsne_results = tsne.fit_transform(data)
    return tsne_results

def plot_tsne(tsne_results, labels):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels, palette="viridis")
    plt.title("t-SNE Visualization of Extracted Features")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Labels")
    plt.savefig("tsne_visualization.png")

def run_PCA(data, n_components=50):
    pca = PCA(n_components=n_components, random_state=42)
    pca_results = pca.fit_transform(data)
    return pca_results

def plot_PCA(pca_results, labels):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=pca_results[:, 0], y=pca_results[:, 1], hue=labels, palette="viridis")
    plt.title("PCA Visualization of Extracted Features")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.legend(title="Labels")
    plt.savefig("pca_visualization.png")

def get_covariance_matrix(data):
    covariance_matrix = np.cov(data, rowvar=False)
    return covariance_matrix


if __name__ == "__main__":
    input_csv_path = "../../../data/preprocessed/task_triples_causal_triple_extraction_overlap_w_winner_type_w_n_winners.csv"
    output_csv_path = "../../../data/preprocessed/hlv_analysis.csv"
    forced = False
    normalized = True

    if not Path(output_csv_path).exists() or forced:
        feature_columns = extract_features(input_csv_path, output_csv_path, normalized)

    df_feature = pd.read_csv(output_csv_path)
    df_raw = pd.read_csv(input_csv_path)
    feature_columns = [x for x in df_feature.columns if x not in df_raw.columns]
    print(f"Extracted features: {feature_columns}")

    X_feature = read_features(output_csv_path, feature_columns, l2_normalize=False)
    pca_results = run_PCA(X_feature, n_components=2)
    #labels = df_raw['n_overlap_triples'].values
    labels = df_raw['n_all_agreed_triples'].values
    labels = ["all agree" if label >= 1 else "disagree" for label in labels]
    #labels = ["high entropy" if label >= 0.5 else "low entropy" for label in labels]

    plot_PCA(pca_results, labels)

    tsne_results = run_tsne(X_feature, metric='cosine')
    plot_tsne(tsne_results, labels)

    covariance_matrix = get_covariance_matrix(pca_results)
    print("Covariance matrix:")
    print(covariance_matrix)
    heatmap = sns.heatmap(covariance_matrix, annot=False)
    figure = heatmap.get_figure()
    figure.savefig('covariance_matrix.png', dpi=400)
