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
        extractor.token_normalize("all")
        extractor.normalize("all")
    feature_columns = [x for x in extractor.data.columns if x not in csv_columns]
    extractor.write_csv(output_csv_path)
    return feature_columns


def read_features(input_csv_path, feature_columns, l2_normalize=False):
    df = pd.read_csv(input_csv_path)
    feature = df[feature_columns].to_numpy()
    print(f"Read features with shape: {feature.shape}")
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
    sns.scatterplot(x=pca_results[:, 1], y=pca_results[:, 2], hue=labels, palette="viridis")
    plt.title("PCA Visualization of Extracted Features")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.legend(title="Labels")
    plt.savefig("pca_visualization.png")

def get_covariance_matrix(data):
    covariance_matrix = np.cov(data, rowvar=False)
    return covariance_matrix


def get_nan_feats(df, feats):
    res = {}
    idx = df.select(feats).select(pl.all().is_nan().any())
    for f in feats:
        if idx[f].sum() > 0:
            res[f] = df[f].is_nan().sum()
    return res

def get_null_feats(df, feats):
    res = {}
    idx = df.select(feats).select(pl.all().is_null().any())
    for f in feats:
        if idx[f].sum() > 0:
            res[f] = df[f].is_null().sum()
    return res

def get_inf_feats(df, feats):
    res = {}
    idx = df.select(feats).select(pl.all().is_infinite().any())
    for f in feats:
        if idx[f].sum() > 0:
            res[f] = df[f].is_infinite().sum()
    return res

def get_features_to_remove(csv_path):
    df = pl.read_csv(csv_path)
    ling_feats = df.columns[10:]
    inf_features = get_inf_feats(df, ling_feats)
    nan_features = get_nan_feats(df, ling_feats)
    null_features = get_null_feats(df, ling_feats)
    features_to_remove = [key for key in inf_features.keys()] + [key for key in nan_features.keys()] + [key for key in null_features.keys()]
    return features_to_remove




if __name__ == "__main__":
    input_csv_path_task_1 = "../../../data/annotated/task_1_annotation.csv"
    input_csv_path_task_2 = "../../../data/preprocessed/task_triples_causal_triple_extraction_overlap_w_winner_type_w_n_winners.csv"
    output_csv_path_task_1 = "../../../data/preprocessed/hlv_analysis_task_1.csv"
    output_csv_path_task_2 = "../../../data/preprocessed/hlv_analysis_task_2.csv"
    forced = False
    normalized = True
    input_csv_path = input_csv_path_task_1
    output_csv_path = output_csv_path_task_1
    length_related_features = ['raw_sequence_length', 'n_tokens', 'n_lemmas', 'n_sentences', 'n_types', 'n_characters', 'n_long_words', 'avg_word_length', 'lemma_token_ratio', 'cttr', 'rttr', 'herdan_c', 'summer_index', 'dugast_u', 'maas_index', 'n_hapax_legomena', 'n_global_lemma_hapax_legomena', 'n_hapax_dislegomena', 'n_global_lemma_hapax_dislegomena', 'n_global_token_hapax_dislegomena', 'hdd', 'sichel_s', 'global_sichel_s', 'giroud_index', 'mtld', 'mattr', 'msttr', 'yule_k', 'simpsons_d', 'herdan_v', 'n_syllables', 'flesch_kincaid_grade', 'n_polysyllables', 'smog', 'ari', 'cli', 'gunning_fog', 'lix', 'rix', 'n_monosyllables', 'n_low_Auditory_sensorimotor', 'n_low_Gustatory_sensorimotor', 'n_low_Haptic_sensorimotor', 'n_low_Interoceptive_sensorimotor', 'n_low_Olfactory_sensorimotor', 'n_low_Visual_sensorimotor', 'n_low_Foot_leg_sensorimotor', 'n_low_Hand_arm_sensorimotor', 'n_low_Head_sensorimotor', 'n_low_Mouth_sensorimotor', 'n_low_Torso_sensorimotor']

    if not Path(output_csv_path).exists() or forced:
        feature_columns = extract_features(input_csv_path, output_csv_path, normalized)

    df_feature = pd.read_csv(output_csv_path)
    df_raw = pd.read_csv(input_csv_path)
    feature_columns = [x for x in df_feature.columns if x not in df_raw.columns]
    features_to_remove = get_features_to_remove(output_csv_path)
    print(f"Extracted features: {len(feature_columns)}")
    print(f"Features to remove due to NaN or Inf values: {len(features_to_remove)}")
    cleaned_feature_columns = [feat for feat in feature_columns if feat not in features_to_remove+length_related_features]
    print(f"Features after cleaning: {cleaned_feature_columns}")
    X_feature = read_features(output_csv_path, cleaned_feature_columns, l2_normalize=False)

    pca_results = run_PCA(X_feature, n_components=5)
    ## task 1 label
    labels = df_raw['dominant_label'].values
    #labels = df_raw['single_winner'].values
    #labels = df_raw['num_unique_labels'].values

    ## task 2 label
    #labels = df_raw['n_overlap_triples'].values
    #labels = df_raw['n_all_agreed_triples'].values
    #labels = ["all agree" if label >= 1 else "disagree" for label in labels]
    #labels = ["high entropy" if label >= 0.5 else "low entropy" for label in labels]

    ## generic label 
    #df_raw['publication_year'] = df_raw['text'].apply(lambda x: x[-26:-21])
    #labels = df_raw['publication_year'].values
    #print(labels)
    #df_raw['word_counts'] = df_raw['text'].str.split().str.len()
    #labels = df_raw['word_counts'].values

    plot_PCA(pca_results, labels)

    tsne_results = run_tsne(X_feature, metric='cosine')
    plot_tsne(tsne_results, labels)

    covariance_matrix = get_covariance_matrix(pca_results)
    print("Covariance matrix:")
    print(covariance_matrix)
    heatmap = sns.heatmap(covariance_matrix, annot=False)
    figure = heatmap.get_figure()
    figure.savefig('covariance_matrix.png', dpi=400)
