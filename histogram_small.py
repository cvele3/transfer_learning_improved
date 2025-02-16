import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)

from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import (
    DeepGraphCNN,
    GCNSupervisedGraphClassification,
    SortPooling,
    GraphConvolution,
)

# --------------------------------------------------------------------
# 1) LOAD PROCESSED DATA AND CV SPLITS
# --------------------------------------------------------------------
PROCESSED_FILE = "processed_data_small.pkl"
with open(PROCESSED_FILE, "rb") as f:
    processed_data = pickle.load(f)

graphs_pep = np.array(processed_data["graphs"])
labels_pep = np.array(processed_data["labels"])
graph_labels = processed_data["graph_labels"]

# Load the CV splits (10 folds)
from data_splits import load_cv_splits
SPLITS_FILE = "cv_splits_small.pkl"
cv_splits = load_cv_splits(SPLITS_FILE)  # list of 10 dicts: {"train_idx", "val_idx", "test_idx"}

model_files = {
    "Baseline Model": "toxicityModel25small_baseline.h5",
    "Freeze GNN": "toxicityModel25peptid_to_small_freezeGNN_folded.h5",
    "Freeze Readout": "toxicityModel25peptid_to_small_freezeReadout_folded.h5",
    "Freeze All, New Output": "toxicityModel25peptid_to_small_freezeAllNewOutput_folded.h5",
}

# --------------------------------------------------------------------
# 2) EVALUATION FUNCTION
# --------------------------------------------------------------------
def evaluate_model(model_path, test_generator, y_true):
    """
    Load a model, predict on test_generator,
    return a dict of evaluation metrics + predicted probabilities.
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
    y_pred_prob = model.predict(test_generator).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_prob)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "y_pred_probs": y_pred_prob,  # keep for histogram
        "y_test": y_true,             # keep for histogram
    }

# --------------------------------------------------------------------
# 3) PREPARE DATA STRUCTURES FOR STORING RESULTS
# --------------------------------------------------------------------
# A PaddedGraphGenerator for test sets
gen = PaddedGraphGenerator(graphs=graphs_pep)

# fold_results will store the metrics for each model across folds.
fold_results = {
    model_name: {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}
    for model_name in model_files
}

# all_predictions will store the combined predictions (and true labels)
# across all folds for each model, so we can make histograms later.
all_predictions = {
    model_name: {"y_test": [], "y_pred_probs": []}
    for model_name in model_files
}

# --------------------------------------------------------------------
# 4) RUN EVALUATION ACROSS ALL FOLDS
# --------------------------------------------------------------------
for fold_i, split in enumerate(cv_splits, start=1):
    print(f"\n--- Fold {fold_i} ---")
    test_idx = split["test_idx"]
    X_test = graphs_pep[test_idx]
    y_test = labels_pep[test_idx]

    # Create test generator for this fold
    test_gen = gen.flow(X_test, y_test, batch_size=32, shuffle=False)

    # Evaluate each model on this fold
    for model_name, model_path in model_files.items():
        result = evaluate_model(model_path, test_gen, y_test)

        # Append metrics
        fold_results[model_name]["accuracy"].append(result["accuracy"])
        fold_results[model_name]["precision"].append(result["precision"])
        fold_results[model_name]["recall"].append(result["recall"])
        fold_results[model_name]["f1"].append(result["f1"])
        fold_results[model_name]["roc_auc"].append(result["roc_auc"])

        # Store predictions for histogram
        all_predictions[model_name]["y_test"].extend(result["y_test"])
        all_predictions[model_name]["y_pred_probs"].extend(result["y_pred_probs"])

# --------------------------------------------------------------------
# 5) PRINT AVERAGE METRICS
# --------------------------------------------------------------------
metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]

print("\n\nEvaluation metrics (averaged over 10 folds):")
for model_name in model_files:
    print(f"\nModel: {model_name}")
    for metric in metric_names:
        values = fold_results[model_name][metric]
        print(f"  {metric}: mean = {np.mean(values):.4f}, std = {np.std(values):.4f}")

# --------------------------------------------------------------------
# 6) GENERATE HISTOGRAMS OF PREDICTED PROBABILITIES FOR EACH MODEL
# --------------------------------------------------------------------
for model_name in model_files:
    print(f"\nGenerating histogram for model: {model_name}")
    # Convert stored lists to NumPy arrays
    y_test_all = np.array(all_predictions[model_name]["y_test"])
    y_pred_probs_all = np.array(all_predictions[model_name]["y_pred_probs"])

    # Binarize using threshold 0.5
    y_pred_labels_all = np.where(y_pred_probs_all >= 0.5, 1, 0)

    # Determine correct/incorrect predictions
    correctly_classified = (y_pred_labels_all == y_test_all)
    incorrectly_classified = ~correctly_classified

    # Define bins and centers
    bins = np.arange(0, 1.05, 0.05)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Histogram counts
    correct_hist, _   = np.histogram(y_pred_probs_all[correctly_classified], bins=bins)
    incorrect_hist, _ = np.histogram(y_pred_probs_all[incorrectly_classified], bins=bins)

    plt.figure(figsize=(10, 6))

    # Overlapping bars
    plt.bar(
        bin_centers,
        correct_hist,
        width=0.04,
        color='blue',
        alpha=0.5,
        label='Correctly Classified',
        edgecolor='black'
    )
    plt.bar(
        bin_centers,
        incorrect_hist,
        width=0.04,
        color='red',
        alpha=0.5,
        label='Incorrectly Classified',
        edgecolor='black'
    )

    # OPTIONAL: place numeric labels stacked one above the other
    offset = 40
    line_spacing = 90
    for x, c_val, i_val in zip(bin_centers, correct_hist, incorrect_hist):
        if c_val == 0 and i_val == 0:
            continue
        top_height = max(c_val, i_val)
        # Red label
        if i_val > 0:
            plt.text(
                x,
                top_height + offset,
                str(i_val),
                ha='center',
                va='bottom',
                color='red',
                fontsize=9
            )
        # Blue label
        if c_val > 0:
            plt.text(
                x,
                top_height + offset + line_spacing,
                str(c_val),
                ha='center',
                va='bottom',
                color='blue',
                fontsize=9
            )

    plt.xlabel("Predicted Toxicity Probability")
    plt.ylabel("Number of Peptides")
    plt.title(f"Histogram of Predicted Toxicity Probabilities (All Folds)\nModel: {model_name}")
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Remove top/right spines if desired
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()
