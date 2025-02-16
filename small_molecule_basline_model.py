# train_model.py

import os
import math
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Import necessary libraries from RDKit and StellarGraph
from rdkit import Chem
from rdkit.Chem import Draw
from stellargraph import StellarGraph
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import LambdaCallback, EarlyStopping
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Import functions for splits
from data_splits import generate_cv_splits, load_cv_splits

######################################
# 1. Load or Generate Preprocessed Data
######################################
PROCESSED_DATA_FILE = "processed_data_small.pkl"

if os.path.exists(PROCESSED_DATA_FILE):
    print("Učitavam prethodno spremljene podatke iz:", PROCESSED_DATA_FILE)
    with open(PROCESSED_DATA_FILE, "rb") as f:
        processed_data = pickle.load(f)
    graphs = processed_data["graphs"]
    labels = processed_data["labels"]
    graph_labels = processed_data["graph_labels"]
    element_to_index = processed_data["element_to_index"]
else:
    print("Ne postoji spremljena datoteka. Pokrećem preprocesiranje podataka...")
    # Read data from Excel
    filepath_raw = 'out.xlsx'
    data_file = pd.read_excel(filepath_raw, header=0, usecols=["SMILES", "HEK"])

    # Initialize an empty list to store tuples
    listOfTuples = []

    # Iterate through each row to extract the SMILES and HEK columns
    for index, row in data_file.iterrows():
        molecule = (row["SMILES"], row["HEK"])
        listOfTuples.append(molecule)

    # Definiraj fiksni vokabular s 27 elemenata
    element_to_index = {
        "N": 0,
        "C": 1,
        "O": 2,
        "F": 3,
        "Cl": 4,
        "S": 5,
        "Na": 6,
        "Br": 7,
        "Se": 8,
        "I": 9,
        "Pt": 10,
        "P": 11,
        "Mg": 12,
        "K": 13,
        "Au": 14,
        "Ir": 15,
        "Cu": 16,
        "B": 17,
        "Zn": 18,
        "Re": 19,
        "Ca": 20,
        "As": 21,
        "Hg": 22,
        "Ru": 23,
        "Pd": 24,
        "Cs": 25,
        "Si": 26,
    }
    NUM_FEATURES = len(element_to_index)
    print("\nFiksni vokabular (27 elemenata) =", element_to_index)

    # (Ostale varijable za normalizaciju se mogu izostaviti ako nisu potrebne.)
    
    # Convert each SMILES into a StellarGraph object
    stellarGraphAllList = []
    ZeroActivity = 0
    OneActivity = 0
    
    for molecule in listOfTuples:
        smileString = molecule[0]
        smileLabel = molecule[1]
        mol = Chem.MolFromSmiles(smileString)
        if mol is None:
            continue  # Skip invalid SMILES
        # Create edge list (both directions)
        edges = []
        for bond in mol.GetBonds():
            edges.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            edges.append((bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))
        
        # Create node features (one-hot encoding using the fixed vocabulary)
        node_features = []
        for atom in mol.GetAtoms():
            elem = atom.GetSymbol()
            if elem not in element_to_index:
                onehot = [0] * NUM_FEATURES
            else:
                onehot = [0] * NUM_FEATURES
                onehot[element_to_index[elem]] = 1
            node_features.append(onehot)
        node_features = np.array(node_features)
        
        # Save edges into a DataFrame (StellarGraph requires a DataFrame for edges)
        edges_df = pd.DataFrame(edges, columns=["source", "target"])
        
        # Create the StellarGraph object
        G = StellarGraph(nodes=node_features, edges=edges_df)
        
        # Optionally, add all examples
        if smileLabel == 1:
            OneActivity += 1
            stellarGraphAllList.append((G, smileLabel))
        elif smileLabel == 0:
            ZeroActivity += 1
            stellarGraphAllList.append((G, smileLabel))
    
    print("Broj primjera za label 0:", ZeroActivity)
    print("Broj primjera za label 1:", OneActivity)
    print("Ukupno primjera:", len(stellarGraphAllList))
    
    # Extract lists of graphs and labels
    graphs = [item[0] for item in stellarGraphAllList]
    labels = [item[1] for item in stellarGraphAllList]
    graph_labels = pd.Series(labels)
    print("Raspodjela labela:")
    print(graph_labels.value_counts().to_frame())
    
    # Save the processed data to a file for future use
    processed_data = {
        "graphs": graphs,
        "labels": labels,
        "graph_labels": graph_labels,
        "element_to_index": element_to_index
    }
    with open(PROCESSED_DATA_FILE, "wb") as f:
        pickle.dump(processed_data, f)
    print("Podaci su spremljeni u", PROCESSED_DATA_FILE)

# Initialize the graph generator using all graphs
generator = PaddedGraphGenerator(graphs=graphs)

######################################
# 2. Load/Generate CV Splits
######################################
SPLITS_FILE = "cv_splits_small.pkl"
if not os.path.exists(SPLITS_FILE):
    cv_splits = generate_cv_splits(graph_labels.values, n_splits=10, val_split=0.2, random_state=42, save_path=SPLITS_FILE)
else:
    cv_splits = load_cv_splits(SPLITS_FILE)

######################################
# 3. Define the Model
######################################
# Model parameters
epochs = 10000
k = 25
layer_sizes = [25, 25, 25, 1]
filter1 = 16
filter2 = 32
filter3 = 128

# Create the DeepGraphCNN model using the same generator
dgcnn_model = DeepGraphCNN(
    layer_sizes=layer_sizes,
    activations=["tanh", "tanh", "tanh", "tanh"],
    k=k,
    bias=False,
    generator=generator,
)
x_inp, x_out = dgcnn_model.in_out_tensors()

x_out = Conv1D(filters=filter1, kernel_size=sum(layer_sizes), strides=sum(layer_sizes))(x_out)
x_out = MaxPool1D(pool_size=2)(x_out)
x_out = Conv1D(filters=filter2, kernel_size=5, strides=1)(x_out)
x_out = Flatten()(x_out)
x_out = Dense(units=filter3, activation="relu")(x_out)
x_out = Dropout(rate=0.2)(x_out)
predictions = Dense(units=1, activation="sigmoid")(x_out)

model = Model(inputs=x_inp, outputs=predictions)
model.compile(optimizer=Adam(lr=0.0001), loss=binary_crossentropy, metrics=["acc"])

# Early stopping callback
callback = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

######################################
# 4. Define Evaluation Functions
######################################
def roc_auc_metric(y_true, y_pred):
    roc_auc = roc_auc_score(y_true, y_pred)
    print("ROC AUC:", roc_auc)
    return roc_auc

def rest_of_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn + 1e-8)
    tnr = tn / (tn + fp + 1e-8)
    gm = math.sqrt(tpr * tnr)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print("GM:", gm)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    return gm, precision, recall, f1

def mcc_metric(y_true, y_pred):
    mcc = matthews_corrcoef(y_true, y_pred)
    print("MCC:", mcc)
    return mcc

######################################
# 5. Cross Validation Training
######################################
histories = []
mcc_values = []
gm_values = []
precision_values = []
recall_values = []
f1_values = []
roc_auc_values = []

# Convert the list of graphs to a numpy array for easier indexing
graphs_arr = np.array(graphs)

all_y_test = []
all_y_pred_probs = []

for fold, split in enumerate(cv_splits):
    print(f"\n--- Fold {fold+1} ---")
    train_idx = split["train_idx"]
    val_idx   = split["val_idx"]
    test_idx  = split["test_idx"]
    
    X_train = graphs_arr[train_idx]
    X_val   = graphs_arr[val_idx]
    X_test  = graphs_arr[test_idx]
    y_train = graph_labels.iloc[train_idx]
    y_val   = graph_labels.iloc[val_idx]
    y_test  = graph_labels.iloc[test_idx]
    
    # Create new generators for this fold
    train_gen = generator.flow(
        X_train,
        targets=y_train,
        batch_size=32,
        symmetric_normalization=False,
    )
    val_gen = generator.flow(
        X_val,
        targets=y_val,
        batch_size=50,
        symmetric_normalization=False,
    )
    test_gen = generator.flow(
        X_test,
        targets=y_test,
        batch_size=50,
        symmetric_normalization=False,
    )
    
    # Train the model (you can reinitialize model per fold if desired)
    history = model.fit(
        train_gen,
        epochs=epochs,
        verbose=1,
        validation_data=val_gen,
        shuffle=True,
        callbacks=[callback]
    )
    histories.append(history)
    
    # Evaluate on the test set
    y_pred_probs = model.predict(test_gen)
    y_pred_probs = np.reshape(y_pred_probs, (-1,))
    roc_auc = roc_auc_metric(y_test, y_pred_probs)
    roc_auc_values.append(roc_auc)
    
    # Spremi rezultate za kasniju analizu
    all_y_test.extend(y_test.to_numpy())
    all_y_pred_probs.extend(y_pred_probs)

    # Use threshold 0.5 for binary predictions
    y_pred = [0 if prob < 0.5 else 1 for prob in y_pred_probs]
    gm, precision, recall, f1 = rest_of_metrics(y_test.to_numpy(), np.array(y_pred))
    mcc = mcc_metric(y_test.to_numpy(), np.array(y_pred))
    
    gm_values.append(gm)
    precision_values.append(precision)
    recall_values.append(recall)
    f1_values.append(f1)
    mcc_values.append(mcc)

# Nakon svih foldova - konvertiraj podatke u numpy array
# Nakon svih foldova - konvertiraj podatke u numpy array
# Nakon svih foldova - konvertiraj podatke u numpy array
all_y_test = np.array(all_y_test)
all_y_pred_probs = np.array(all_y_pred_probs)

# Generiranje binarnih predikcija (klasifikacija na temelju praga 0.5)
all_y_pred_labels = np.where(all_y_pred_probs >= 0.5, 1, 0)

# Određivanje točno i pogrešno klasificiranih instanci
correctly_classified = (all_y_pred_labels == all_y_test)
incorrectly_classified = ~correctly_classified

# Definiraj binove
bins = np.arange(0, 1.05, 0.05)
bin_centers = (bins[:-1] + bins[1:]) / 2

# Kreiraj histogram podatke
correct_hist, _   = np.histogram(all_y_pred_probs[correctly_classified], bins=bins)
incorrect_hist, _ = np.histogram(all_y_pred_probs[incorrectly_classified], bins=bins)

plt.figure(figsize=(10, 6))

# Overlapping histograms: same x-centers, same width
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

offset = 30         # vertical offset above the top bar
line_spacing = 40  # extra space between the two labels

for x, c, i in zip(bin_centers, correct_hist, incorrect_hist):
    # Skip if no small molecules in this bin
    if c == 0 and i == 0:
        continue

    # We'll place both labels above whichever bar is taller,
    # but the correct label (blue) will always be on top, and
    # the incorrect label (red) just below it, separated by line_spacing.
    top_height = max(c, i)

    # Red label (lower line, even if c < i or c > i)
    if i > 0:
        plt.text(
            x,
            top_height + offset,   # base position for the lower label
            str(i),
            ha='center',
            va='bottom',
            color='red',
            fontsize=9
        )

    # Blue label (upper line, always above the red label by line_spacing)
    if c > 0:
        plt.text(
            x,
            top_height + offset + line_spacing,  # further up
            str(c),
            ha='center',
            va='bottom',
            color='blue',
            fontsize=9
        )

plt.xlabel("Predicted Toxicity Probability")
plt.ylabel("Number of Small molecules")
# plt.title("Histogram of Predicted Toxicity Probabilities (All 10 Folds)")
plt.xticks(np.arange(0, 1.1, 0.1))
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
ax = plt.gca()  # Get current axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

# Save the final trained model (or the one from the last fold)
# model.save('toxicityModel25small_baseline.h5')

######################################
# 6. Save and Plot Results
######################################
# Evaluate final model on test generator (from last fold)
test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print(f"\t{name}: {val:0.4f}")

# Create a DataFrame with evaluation metrics
# data = {
#     "Metric": ["MCC", "GM", "Precision", "Recall", "F1", "ROC AUC"],
#     "Average": [np.mean(mcc_values), np.mean(gm_values), np.mean(precision_values), 
#                 np.mean(recall_values), np.mean(f1_values), np.mean(roc_auc_values)],
#     "Maximum": [np.max(mcc_values), np.max(gm_values), np.max(precision_values), 
#                 np.max(recall_values), np.max(f1_values), np.max(roc_auc_values)],
#     "Minimum": [np.min(mcc_values), np.min(gm_values), np.min(precision_values), 
#                 np.min(recall_values), np.min(f1_values), np.min(roc_auc_values)]
# }
# df_metrics = pd.DataFrame(data)
# excel_save = f"metrics_small_baseline_k{k}_{layer_sizes}_with_{filter1}-{filter2}-{filter3}.xlsx"
# df_metrics.to_excel(excel_save, index=False)
# print(f"Rezultati su spremljeni u '{excel_save}'.")
