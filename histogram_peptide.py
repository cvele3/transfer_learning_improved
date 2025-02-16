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
PROCESSED_FILE = "processed_data.pkl"
with open(PROCESSED_FILE, "rb") as f:
    processed_data = pickle.load(f)

graphs_pep = np.array(processed_data["graphs"])
labels_pep = np.array(processed_data["labels"])
graph_labels = processed_data["graph_labels"]

# Load the CV splits (10 folds)
from data_splits import load_cv_splits
SPLITS_FILE = "cv_splits.pkl"
cv_splits = load_cv_splits(SPLITS_FILE)  # list of 10 dicts with keys: "train_idx", "val_idx", "test_idx"

# Define your model files
model_files = {
    "Baseline Model": "toxicityModel25peptide_baseline.h5",
    "Freeze GNN": "toxicityModel25small_to_peptide_freezeGNN_folded.h5",
    "Freeze Readout": "toxicityModel25small_to_peptide_freezeReadout_folded.h5",
    "Freeze All, New Output": "toxicityModel25small_to_peptide_freezeAllNewOutput_folded.h5",
}

# --------------------------------------------------------------------
# 2) DEFINE EVALUATION FUNCTION
# --------------------------------------------------------------------
def evaluate_model(model_path, test_generator, y_true):
    """
    Load a model, predict on test_generator,
    and return a dict of evaluation metrics plus predicted probabilities.
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
        "y_pred_probs": y_pred_prob,  # for histogram generation
        "y_test": y_true,             # for histogram generation
    }

# --------------------------------------------------------------------
# 3) SET UP DATA STRUCTURES FOR STORING RESULTS
# --------------------------------------------------------------------
# Create a PaddedGraphGenerator for the entire dataset
gen = PaddedGraphGenerator(graphs=graphs_pep)

# fold_results stores evaluation metrics for each model across folds
fold_results = {
    model_name: {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}
    for model_name in model_files
}

# all_predictions stores the raw predictions and true labels for histogram plotting
all_predictions = {
    model_name: {"y_test": [], "y_pred_probs": []}
    for model_name in model_files
}

# --------------------------------------------------------------------
# 4) RUN EVALUATION ACROSS ALL 10 FOLDS
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

        # Append evaluation metrics for the current fold
        fold_results[model_name]["accuracy"].append(result["accuracy"])
        fold_results[model_name]["precision"].append(result["precision"])
        fold_results[model_name]["recall"].append(result["recall"])
        fold_results[model_name]["f1"].append(result["f1"])
        fold_results[model_name]["roc_auc"].append(result["roc_auc"])

        # Save predictions for histogram generation
        all_predictions[model_name]["y_test"].extend(result["y_test"])
        all_predictions[model_name]["y_pred_probs"].extend(result["y_pred_probs"])

# Print average metrics over folds
print("\n\nEvaluation metrics (averaged over 10 folds):")
for model_name in model_files:
    print(f"\nModel: {model_name}")
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        values = fold_results[model_name][metric]
        print(f"  {metric}: mean = {np.mean(values):.4f}, std = {np.std(values):.4f}")

# --------------------------------------------------------------------
# 5) GENERATE HISTOGRAMS OF PREDICTED PROBABILITIES FOR EACH MODEL
# --------------------------------------------------------------------
# For each model, we combine predictions from all folds and plot an overlapping histogram
for model_name in model_files:
    print(f"\nGenerating histogram for model: {model_name}")
    # Convert stored lists to NumPy arrays
    y_test_all = np.array(all_predictions[model_name]["y_test"])
    y_pred_probs_all = np.array(all_predictions[model_name]["y_pred_probs"])

    # Binarize predictions using threshold 0.5
    y_pred_labels_all = np.where(y_pred_probs_all >= 0.5, 1, 0)

    # Determine correct vs. incorrect predictions
    correctly_classified = (y_pred_labels_all == y_test_all)
    incorrectly_classified = ~correctly_classified

    # Define bins and calculate bin centers
    bins = np.arange(0, 1.05, 0.05)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Generate histogram counts for correct and incorrect predictions
    correct_hist, _   = np.histogram(y_pred_probs_all[correctly_classified], bins=bins)
    incorrect_hist, _ = np.histogram(y_pred_probs_all[incorrectly_classified], bins=bins)

    plt.figure(figsize=(10, 6))

    # Draw overlapping bars for correct (blue) and incorrect (red) predictions
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

    # OPTIONAL: Add numeric labels above the bars
    offset = 40         # vertical offset above the taller bar
    line_spacing = 90   # additional spacing between labels

    for x, c_val, i_val in zip(bin_centers, correct_hist, incorrect_hist):
        if c_val == 0 and i_val == 0:
            continue
        top_height = max(c_val, i_val)
        # Place red label (incorrect) on the lower line
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
        # Place blue label (correct) on the upper line
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

    # Remove top and right spines to avoid clashing with labels
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()
