import math
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import rdkit.Chem
import networkx as nx
import matplotlib.pyplot as plt
import stellargraph as sg

from rdkit import Chem
from rdkit.Chem import Draw

from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import matthews_corrcoef, precision_recall_curve

from stellargraph import StellarGraph, datasets
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import PaddedGraphGenerator, GraphSAGELinkGenerator
from stellargraph.layer import DeepGraphCNN, GraphSAGE, link_classification

from tensorflow.keras import Model, optimizers, losses, metrics
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.utils import Sequence
import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPool1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split

# Ako je potrebno za custom slojeve iz StellarGrapha:
from stellargraph.layer import DeepGraphCNN
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph import StellarGraph

from rdkit import Chem

# Uvoz ovih slojeva ako ih treba prilikom load_model
from stellargraph.layer import (
    DeepGraphCNN,
    GCNSupervisedGraphClassification,
    SortPooling,
    GraphConvolution,
)
from stellargraph.layer.graph_classification import SortPooling

from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
import tensorflow.keras.backend as K



filepath_raw = 'combined.xlsx'
data_file = pd.read_excel(filepath_raw, header=0, usecols=["TOXICITY", "SMILES", "TYPE"])

listOfTuples = []

data_file.reset_index()
for index, row in data_file.iterrows():
    smiles = row['SMILES']
    label = row["TOXICITY"]
    type = row["TYPE"]
    molecule = (row["SMILES"], row["TOXICITY"], row["TYPE"])
    listOfTuples.append(molecule)



ZeroActivity = 0
OneActivity = 0
stellarGraphAllList = []

from collections import Counter

# Get a list of all unique elements in the molecules and extract features.
# At the same time, create a list of seen properties from each atom.
all_elements = []
lst_degree = []
lst_formal_charge = []
lst_radical_electrons = []
lst_hybridization = []
lst_aromatic = []
for molecule in listOfTuples:
    mol = Chem.MolFromSmiles(molecule[0])
    atoms = mol.GetAtoms()
    for atom in atoms:
        lst_degree.append(atom.GetDegree())
        lst_formal_charge.append(atom.GetFormalCharge())
        lst_radical_electrons.append(atom.GetNumRadicalElectrons())
        lst_hybridization.append(atom.GetHybridization().real)
        lst_aromatic.append(atom.GetIsAromatic())

        element = atom.GetSymbol()
        if element not in all_elements:
            all_elements.append(element)

# Determine min and max values for each property/feature.
# This values will be later used for min-max scaling.
min_degree, max_degree = min(lst_degree), max(lst_degree)
min_formal_charge, max_formal_charge = min(lst_formal_charge), max(lst_formal_charge)
min_radical_electrons, max_radical_electrons = min(lst_radical_electrons), max(lst_radical_electrons)
min_hybridization, max_hybridization = min(lst_hybridization), max(lst_hybridization)
min_aromatic, max_aromatic = min(lst_aromatic), max(lst_aromatic)

# Create a dictionary that maps each element to a unique index
element_to_index = {element: index for index, element in enumerate(all_elements)}

for molecule in listOfTuples:

    smileString = molecule[0]
    smileLabel = molecule[1]
    type = molecule[2]

    # Convert the SMILES string into a molecular graph using RDKit
    mol = Chem.MolFromSmiles(smileString)
    atoms = mol.GetAtoms()
    edges = []
    for bond in mol.GetBonds():
        edges.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        edges.append((bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))

    # Convert the RDKit atom objects to a Numpy array of node features
    node_features = []
    for atom in atoms:
        element = atom.GetSymbol()
        degree = (atom.GetDegree() - min_degree) / (max_degree - min_degree)
        # formal_charge = (atom.GetFormalCharge() - min_formal_charge) / (max_formal_charge - min_formal_charge)
        # num_radical_electrons = (atom.GetNumRadicalElectrons() - min_radical_electrons) / (max_radical_electrons - min_radical_electrons)
        hybridization = (atom.GetHybridization().real - min_hybridization) / (max_hybridization - min_hybridization)
        aromatic = (atom.GetIsAromatic() - min_aromatic) / (max_aromatic - min_aromatic)
        element_onehot = [0] * len(all_elements)
        element_onehot[element_to_index[element]] = 1
        node_features.append(element_onehot)
        #node_features.append(element_onehot + [degree, hybridization, aromatic])
    node_features = np.array(node_features)

    # Convert the edges to a pandas DataFrame
    edges_df = pd.DataFrame(edges, columns=["source", "target"])

    # Create a StellarGraph object from the molecular graph
    G = StellarGraph(nodes=node_features, edges=edges_df)

    # if smileLabel == 1 and OneActivity < 1000:
    if smileLabel == 1:
        OneActivity += 1
        skup = (G, smileLabel, type)
        stellarGraphAllList.append(skup)

    # if smileLabel == 0 and ZeroActivity < 1000:
    if smileLabel == 0:
        ZeroActivity += 1
        skup = (G, smileLabel, type)
        stellarGraphAllList.append(skup)

graphs = []
labels = []
types = []

for triple in stellarGraphAllList:
    grafinjo = triple[0]
    active = triple[1]
    graphs.append(grafinjo)
    labels.append(active)
    types.append(triple[2])


graph_labels = pd.Series(labels)

generator = PaddedGraphGenerator(graphs=graphs)




epochs = 10000

# Define the number of rows for the output tensor and the layer sizes
k = 25
layer_sizes = [25, 25, 25, 1]
filter1 = 16
filter2 = 32
filter3 = 128

# Create the DeepGraphCNN model
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

# Create the model and compile it
model = Model(inputs=x_inp, outputs=predictions)


gm_values = []
precision_values = []
recall_values = []
f1_values = []
roc_auc_values = []
fpr_values = []
tpr_values = []


def roc_auc_metric(y_true, y_pred):
    y_pred = K.cast(K.round(y_pred), K.floatx())
    y_true = K.cast(y_true, K.floatx())

    y_pred_np = K.eval(y_pred)  # Convert y_pred tensor to NumPy array

    roc_auc = roc_auc_score(y_true, y_pred)
    print("ROC AUC: ", roc_auc)
    roc_auc_values.append(K.get_value(roc_auc))


def rest_of_metrics(y_true, y_pred):
    y_pred = K.cast(K.round(y_pred), K.floatx())
    y_true = K.cast(y_true, K.floatx())

    y_pred_np = K.eval(y_pred)  # Convert y_pred tensor to NumPy array

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_np).ravel()

    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    gm = math.sqrt(tpr * tnr)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("GM: ", gm)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)

    gm_values.append(K.get_value(gm))
    precision_values.append(K.get_value(precision))
    recall_values.append(K.get_value(recall))
    f1_values.append(K.get_value(f1))
    fpr_values.append(K.get_value(fpr))
    tpr_values.append(K.get_value(tpr))


restOfMetrics_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: metrics(test_gen.targets, model.predict(test_gen)))

mcc_values = []


# Create a LambdaCallback to calculate MCC at the end of each epoch
def mcc_metric(y_true, y_pred):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_pred = y_pred.round()
    mcc = matthews_corrcoef(y_true, y_pred)
    print("MCC: ", mcc)
    mcc_values.append(mcc)


mcc_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: mcc_metric(test_gen.targets, model.predict(test_gen)))
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

model.compile(
    optimizer=Adam(lr=0.0001),
    loss=binary_crossentropy,
    metrics=["acc"]
)


# Convert graphs to a NumPy array if not already done.
graphs = np.array(graphs)
# Convert labels to a pandas Series (or use a NumPy array) for indexing.
graph_labels = pd.Series(labels)

# First, separate indices by molecule type.
small_molecule_indices = [i for i, t in enumerate(types) if t == "small"]
peptide_indices = [i for i, t in enumerate(types) if t == "peptid"]

# Extract only the small molecule subset (and their labels) for cross-validation splitting.
small_graphs = graphs[small_molecule_indices]
small_labels = graph_labels.iloc[small_molecule_indices]

# Set up stratified K-Fold splitting on small molecules only.
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
histories = []

for fold, (sm_train_idx, sm_test_idx) in enumerate(skf.split(small_graphs, small_labels)):
    print(f"Processing Fold {fold + 1}...")
    
    # Map the indices from the small molecule subset back to the original indices.
    sm_train_indices = np.array(small_molecule_indices)[sm_train_idx]
    sm_test_indices = np.array(small_molecule_indices)[sm_test_idx]
    
    # For this fold, the test set is exclusively the selected small molecules.
    X_test = graphs[sm_test_indices]
    y_test = graph_labels.iloc[sm_test_indices]
    
    # The training set is the union of:
    #   - all peptides (which you always want to train on), and
    #   - the remaining small molecules (those not chosen for testing in this fold).
    fold_train_indices = np.concatenate((peptide_indices, sm_train_indices))
    X_train_val = graphs[fold_train_indices]
    y_train_val = graph_labels.iloc[fold_train_indices]
    
    # (Optional) Further split the training data into training and validation sets.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )
    
    # Create a PaddedGraphGenerator based on all graphs (needed by StellarGraph).
    gen = PaddedGraphGenerator(graphs=graphs)
    
    # Create data generators for training, validation, and testing.
    train_gen = gen.flow(X_train, targets=y_train, batch_size=32, symmetric_normalization=False)
    val_gen = gen.flow(X_val, targets=y_val, batch_size=32, symmetric_normalization=False)
    test_gen = gen.flow(X_test, targets=y_test, batch_size=32, symmetric_normalization=False)
    
    # Train the model on the training data and validate on the validation data.
    history = model.fit(
        train_gen,
        epochs=epochs,
        verbose=1,
        validation_data=val_gen,
        shuffle=True,
        callbacks=[callback]  # Assume "callback" (e.g., early stopping) is defined.
    )
    histories.append(history)
    
    # Evaluate on the test set (which contains only small molecules).
    y_pred = model.predict(test_gen)
    y_pred = np.reshape(y_pred, (-1,))
    roc_auc_metric(y_test, y_pred)
    # Convert probabilities to binary predictions.
    y_pred_binary = [0 if prob < 0.5 else 1 for prob in y_pred]
    
    # Convert test labels to a NumPy array for metric functions.
    y_test_np = y_test.to_numpy()
    y_test_np = np.reshape(y_test_np, (-1,))
    
    # Calculate additional metrics.
    rest_of_metrics(y_test_np, y_pred_binary)
    mcc_metric(y_test_np, y_pred_binary)

# Save the final model after cross-validation.
model.save('combinedModel25smallbetter.h5')

test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))


data = {
    "Metric": ["MCC", "GM", "Precision", "Recall", "F1", "ROC AUC", "FPR", "TPR"],
    "Average": [np.mean(mcc_values), np.mean(gm_values), np.mean(precision_values), np.mean(recall_values),
                np.mean(f1_values), np.mean(roc_auc_values), np.mean(fpr_values), np.mean(tpr_values)],
    "Maximum": [np.max(mcc_values), np.max(gm_values), np.max(precision_values), np.max(recall_values),
                np.max(f1_values), np.max(roc_auc_values), np.max(fpr_values), np.max(tpr_values)],
    "Minimum": [np.min(mcc_values), np.min(gm_values), np.min(precision_values), np.min(recall_values),
                np.min(f1_values), np.min(roc_auc_values), np.min(fpr_values), np.min(tpr_values)]
}

# Create a pandas DataFrame from the dictionary
df = pd.DataFrame(data)


filename = f"metrics_smallbetter_test_k{k}_layers_{layer_sizes}_filters_{filter1}-{filter2}-{filter3}.xlsx"

# Save the DataFrame to an Excel file
df.to_excel(filename, index=False)