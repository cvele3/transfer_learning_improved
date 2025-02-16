import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import (
    DeepGraphCNN,
    GCNSupervisedGraphClassification,
    SortPooling,
    GraphConvolution,
)

# --------------------------------------------------------------------
# Load processed data from file (processed_data_small.pkl)
# --------------------------------------------------------------------
PROCESSED_FILE = "processed_data.pkl"
with open(PROCESSED_FILE, "rb") as f:
    processed_data = pickle.load(f)

# Assume processed_data contains: "graphs", "labels", "graph_labels"
graphs_pep = np.array(processed_data["graphs"])
labels_pep = np.array(processed_data["labels"])
graph_labels = processed_data["graph_labels"]

# Import your CV split function from data_splits.py
from data_splits import load_cv_splits

# Load the CV splits file (generated using data_splits.py)
SPLITS_FILE = "cv_splits.pkl"
splits = load_cv_splits(SPLITS_FILE)  # Expecting a list of 10 dicts with keys: "train_idx", "val_idx", "test_idx"

# List your model file paths
model_files = {
    "Baseline Model": "toxicityModel25peptide_baseline.h5",
    "Freeze GNN": "toxicityModel25small_to_peptide_freezeGNN_folded.h5",
    "Freeze Readout": "toxicityModel25small_to_peptide_freezeReadout_folded.h5",
    "Freeze All, New Output": "toxicityModel25small_to_peptide_freezeAllNewOutput_folded.h5",
    # "Combined Model": "combinedModel25peptidebetter.h5",
}

def evaluate_model(model_path, test_generator, y_true):
    """
    Load a model from model_path, predict on test_generator,
    and return a dictionary of evaluation metrics.
    """
    model = load_model(
        model_path,
        custom_objects={
            "DeepGraphCNN": DeepGraphCNN,
            "GCNSupervisedGraphClassification": GCNSupervisedGraphClassification,
            "SortPooling": SortPooling,
            "GraphConvolution": GraphConvolution,
        }
    )
    y_pred_prob = model.predict(test_generator)
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_prob)
    
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc_auc}

# Create a PaddedGraphGenerator using all your graphs (for consistency)
gen = PaddedGraphGenerator(graphs=graphs_pep)

# Prepare a dictionary to store metrics for each model across the 10 folds
fold_results = {model_name: {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}
                for model_name in model_files.keys()}

# Iterate over all 10 folds and evaluate each model on that fold's test data
for fold in splits:
    test_idx = fold["test_idx"]
    X_test = graphs_pep[test_idx]
    y_test = labels_pep[test_idx]
    
    # Create a test generator for this fold
    test_gen = gen.flow(X_test, y_test, batch_size=32, shuffle=False)
    
    # Evaluate each model on this fold's test set and store the metrics
    for model_name, model_path in model_files.items():
        metrics = evaluate_model(model_path, test_gen, y_test)
        for metric_name, value in metrics.items():
            fold_results[model_name][metric_name].append(value)

# --------------------------------------------------------------------
# For each metric, perform Friedman test and, if significant, run Dunn's post hoc test,
# create box-and-whisker plots, and check which model is significantly better.
# --------------------------------------------------------------------
metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
friedman_results = {}

print("Evaluation metrics (averaged over 10 folds):")
for model_name in model_files.keys():
    print(f"\nModel: {model_name}")
    for metric in metric_names:
        values = fold_results[model_name][metric]
        print(f"  {metric}: mean = {np.mean(values):.4f}, std = {np.std(values):.4f}")

print("\nFriedman Test Results and Post Hoc Analysis:")
for metric in metric_names:
    # Create a DataFrame where each column is a model and rows are fold values
    data_for_metric = {model_name: fold_results[model_name][metric] for model_name in model_files.keys()}
    df_metric = pd.DataFrame(data_for_metric)
    
    # Perform the Friedman test
    data_list = [df_metric[col].values for col in df_metric.columns]
    stat, p_value = friedmanchisquare(*data_list)
    friedman_results[metric] = {"statistic": stat, "p_value": p_value}
    print(f"\nMetric: {metric}")
    print(f"Friedman chi-square = {stat:.4f}, p-value = {p_value:.4f}")
    
    # Create a boxplot for the metric
    df_melted = df_metric.melt(var_name="Model", value_name=metric)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Model", y=metric, data=df_melted)
    plt.title(f"Box-and-Whisker Plot for {metric}")
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.show()
    
    # If Friedman test is significant, perform Dunn's post hoc test and check which model is better
    if p_value < 0.05:
        # Convert the wide format DataFrame to long format (if not already done)
        df_melted = df_metric.melt(var_name="Model", value_name=metric)
        # Perform Dunn's post hoc test specifying the value and group columns
        dunn_results = sp.posthoc_dunn(df_melted, val_col=metric, group_col="Model", p_adjust='bonferroni')
        print(f"\nPost Hoc Dunn Test Results for {metric}:")
        print(pd.DataFrame(dunn_results, index=df_metric.columns, columns=df_metric.columns))
        
        # Determine the best model by mean
        means = {model: np.mean(fold_results[model][metric]) for model in model_files.keys()}
        best_model = max(means, key=means.get)
        best_mean = means[best_model]
        
        # Check if best model is significantly better than every other model
        is_significantly_better = True
        for model in model_files.keys():
            if model == best_model:
                continue
            # Look up the p-value for the pairwise comparison
            p_val_comparison = dunn_results.loc[best_model, model]
            if p_val_comparison >= 0.05:
                is_significantly_better = False
                break
        
        if is_significantly_better:
            print(f"\nFor metric '{metric}', the model '{best_model}' is significantly better than all others (mean = {best_mean:.4f}).")
        else:
            print(f"\nFor metric '{metric}', although '{best_model}' has the best mean (mean = {best_mean:.4f}), it is not significantly better than all other models.")
    else:
        print(f"No significant difference for {metric} (p-value >= 0.05); post hoc test not performed.")
