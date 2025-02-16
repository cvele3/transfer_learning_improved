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
from tensorflow.keras.callbacks import LambdaCallback, EarlyStopping
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

from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Uvoz funkcija za generiranje/učitavanje CV splitova (cv_splits.pkl)
from data_splits import generate_cv_splits, load_cv_splits


##############################################################
# 1. Učitavanje podataka i generiranje StellarGraph objekata
##############################################################
filepath_raw = 'out.xlsx'
data_file = pd.read_excel(filepath_raw, header=0, usecols=["SMILES", "HEK"])

# Initialize an empty list to store tuples
listOfTuples = []

# Iterate through each row to extract the SMILES and HEK columns
for index, row in data_file.iterrows():
    molecule = (row["SMILES"], row["HEK"])
    listOfTuples.append(molecule)

print("Broj peptidnih zapisa:", len(listOfTuples))

# Koristimo fiksni vokabular s 27 elemenata
element_to_index = {
    "N": 0, "C": 1, "O": 2, "F": 3, "Cl": 4, "S": 5, "Na": 6, "Br": 7,
    "Se": 8, "I": 9, "Pt": 10, "P": 11, "Mg": 12, "K": 13, "Au": 14,
    "Ir": 15, "Cu": 16, "B": 17, "Zn": 18, "Re": 19, "Ca": 20, "As": 21,
    "Hg": 22, "Ru": 23, "Pd": 24, "Cs": 25, "Si": 26,
}
NUM_FEATURES = len(element_to_index)
print("\nFiksni vokabular (27 elemenata) =", element_to_index)

stellarGraphAllList = []
ZeroActivity = 0
OneActivity = 0

for molecule in listOfTuples:
    smileString = molecule[0]
    smileLabel = molecule[1]
    mol = Chem.MolFromSmiles(smileString)
    if mol is None:
        continue  # Preskoči nevalidne SMILES
    # Generiraj bridove (oba smjera)
    edges = []
    for bond in mol.GetBonds():
        edges.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        edges.append((bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))
    # Generiraj node features (one-hot kodiranje)
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
    
    edges_df = pd.DataFrame(edges, columns=["source", "target"])
    G = StellarGraph(nodes=node_features, edges=edges_df)
    
    if smileLabel == 1:
        OneActivity += 1
        stellarGraphAllList.append((G, smileLabel))
    elif smileLabel == 0:
        ZeroActivity += 1
        stellarGraphAllList.append((G, smileLabel))

print("Broj primjera za label 0:", ZeroActivity)
print("Broj primjera za label 1:", OneActivity)
print("Ukupno primjera:", len(stellarGraphAllList))

graphs = [item[0] for item in stellarGraphAllList]
labels = [item[1] for item in stellarGraphAllList]
graph_labels = pd.Series(labels)
print("Distribucija labela:\n", graph_labels.value_counts().to_frame())

# Inicijaliziraj generator koristeći sve grafove
generator = PaddedGraphGenerator(graphs=graphs)

##############################################################
# 2. Učitavanje ili generiranje CV splitova (cv_splits.pkl)
##############################################################
SPLITS_FILE = "cv_splits_small.pkl"
if not os.path.exists(SPLITS_FILE):
    cv_splits = generate_cv_splits(graph_labels.values, n_splits=10, val_split=0.2, random_state=42, save_path=SPLITS_FILE)
else:
    cv_splits = load_cv_splits(SPLITS_FILE)

##############################################################
# 3. Transfer Learning – Učitavanje baseline modela i dotreniranje
##############################################################
# PRETRAINED_MODEL_PATH je putanja do baseline modela (npr. toxicityModel25small.h5)
PRETRAINED_MODEL_PATH = "toxicityModel25peptide_baseline.h5"

# Funkcija za učitavanje baseline modela
def load_pretrained_model():
    model_loaded = load_model(
        PRETRAINED_MODEL_PATH,
        custom_objects={
            "DeepGraphCNN": DeepGraphCNN,
            "GCNSupervisedGraphClassification": GCNSupervisedGraphClassification,
            "SortPooling": SortPooling,
            "GraphConvolution": GraphConvolution,
        }
    )
    return model_loaded

# Funkcije za evaluaciju
def roc_auc_metric(y_true, y_pred):
    roc_auc = roc_auc_score(y_true, y_pred)
    print("ROC AUC:", roc_auc)
    return roc_auc

def rest_of_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, (y_pred >= 0.5).astype(int)).ravel()
    tpr = tp / (tp + fn + 1e-8)
    tnr = tn / (tn + fp + 1e-8)
    gm = math.sqrt(tpr * tnr)
    precision = precision_score(y_true, (y_pred >= 0.5).astype(int))
    recall = recall_score(y_true, (y_pred >= 0.5).astype(int))
    f1 = f1_score(y_true, (y_pred >= 0.5).astype(int))
    print("GM:", gm)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    return gm, precision, recall, f1

def mcc_metric(y_true, y_pred):
    mcc = matthews_corrcoef(y_true, (y_pred >= 0.5).astype(int))
    print("MCC:", mcc)
    return mcc

##############################################################
# 4. Transfer Learning – dotreniranje modela pomoću CV splitova
##############################################################
epochs = 10000
# Set EarlyStopping callback
callback = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)



# Za svaki CV split koristimo iste podjele (učitane iz cv_splits.pkl)
# Metoda 1: Zamrzavanje GNN slojeva
print("\n=== METODA 1: Zamrzavanje GNN slojeva + 10-fold CV ===")
fold_index = 0
for split in cv_splits:
    fold_index += 1
    print(f"\n--- Fold {fold_index} ---")
    model1 = load_pretrained_model()
    # Zamrzavanje slojeva koji sadrže "deep_graph_cnn" ili "graph_conv"
    for layer in model1.layers:
        if "deep_graph_cnn" in layer.name or "graph_conv" in layer.name:
            layer.trainable = False
        else:
            layer.trainable = True
    model1.compile(optimizer=Adam(learning_rate=1e-4), loss=binary_crossentropy, metrics=["accuracy"])
    
    graphs_arr = np.array(graphs)
    labels_arr = np.array(graph_labels)
    X_train = graphs_arr[split["train_idx"]]
    X_val   = graphs_arr[split["val_idx"]]
    X_test  = graphs_arr[split["test_idx"]]
    y_train = labels_arr[split["train_idx"]]
    y_val   = labels_arr[split["val_idx"]]
    y_test  = labels_arr[split["test_idx"]]
    
    # Dodatno razdvajanje trening skupa na trening i validaciju (ako je potrebno)
    X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=42)
    y_train, y_val = train_test_split(y_train, test_size=0.2, random_state=42)
    
    gen_fold = PaddedGraphGenerator(graphs=graphs_arr)
    train_gen = gen_fold.flow(X_train, y_train, batch_size=32, shuffle=True)
    val_gen   = gen_fold.flow(X_val,   y_val,   batch_size=32, shuffle=False)
    test_gen  = gen_fold.flow(X_test,  y_test,  batch_size=32, shuffle=False)
    
    history1 = model1.fit(train_gen, validation_data=val_gen, epochs=epochs, verbose=1, shuffle=True, callbacks=[callback])
    
    y_pred = model1.predict(test_gen)
    y_pred = np.reshape(y_pred, (-1,))
    roc_auc_metric(y_test, y_pred)
    rest_of_metrics(y_test, y_pred)
    mcc_metric(y_test, y_pred)

model1.save("toxicityModel25peptid_to_small_freezeGNN_folded.h5")

# Metoda 2: Zamrzavanje READOUT/dense slojeva
print("\n=== METODA 2: Zamrzavanje READOUT/dense slojeva + 10-fold CV ===")
fold_index = 0
for split in cv_splits:
    fold_index += 1
    print(f"\n--- Fold {fold_index} ---")
    model2 = load_pretrained_model()
    # Zamrzavanje slojeva koji sadrže "dense", "dropout", "flatten" ili "readout"
    for layer in model2.layers:
        if any(x in layer.name for x in ["dense", "dropout", "flatten", "readout"]):
            layer.trainable = False
        else:
            layer.trainable = True
    model2.compile(optimizer=Adam(learning_rate=1e-5), loss=binary_crossentropy, metrics=["accuracy"])
    
    graphs_arr = np.array(graphs)
    labels_arr = np.array(graph_labels)
    X_train = graphs_arr[split["train_idx"]]
    X_val   = graphs_arr[split["val_idx"]]
    X_test  = graphs_arr[split["test_idx"]]
    y_train = labels_arr[split["train_idx"]]
    y_val   = labels_arr[split["val_idx"]]
    y_test  = labels_arr[split["test_idx"]]
    
    X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=42)
    y_train, y_val = train_test_split(y_train, test_size=0.2, random_state=42)
    
    gen_fold = PaddedGraphGenerator(graphs=graphs_arr)
    train_gen = gen_fold.flow(X_train, y_train, batch_size=32, shuffle=True)
    val_gen   = gen_fold.flow(X_val,   y_val,   batch_size=32, shuffle=False)
    test_gen  = gen_fold.flow(X_test,  y_test,  batch_size=32, shuffle=False)
    
    history2 = model2.fit(train_gen, validation_data=val_gen, epochs=epochs, verbose=1, shuffle=True, callbacks=[callback])
    
    y_pred = model2.predict(test_gen)
    y_pred = np.reshape(y_pred, (-1,))
    roc_auc_metric(y_test, y_pred)
    rest_of_metrics(y_test, y_pred)
    mcc_metric(y_test, y_pred)

model2.save("toxicityModel25peptid_to_small_freezeReadout_folded.h5")

# Metoda 3: Zamrzavanje svih slojeva + novi izlazni sloj
print("\n=== METODA 3: Zamrzavanje svih slojeva + novi izlazni sloj + 10-fold CV ===")
fold_index = 0
for split in cv_splits:
    fold_index += 1
    print(f"\n--- Fold {fold_index} ---")
    base_model = load_pretrained_model()
    for layer in base_model.layers:
        layer.trainable = False
    # Dodaj novi izlaz
    intermediate_output = base_model.layers[-2].output
    new_output = Dense(1, activation="sigmoid", name="new_output")(intermediate_output)
    model3 = Model(inputs=base_model.input, outputs=new_output)
    model3.compile(optimizer=Adam(learning_rate=1e-4), loss=binary_crossentropy, metrics=["accuracy"])
    
    graphs_arr = np.array(graphs)
    labels_arr = np.array(graph_labels)
    X_train = graphs_arr[split["train_idx"]]
    X_val   = graphs_arr[split["val_idx"]]
    X_test  = graphs_arr[split["test_idx"]]
    y_train = labels_arr[split["train_idx"]]
    y_val   = labels_arr[split["val_idx"]]
    y_test  = labels_arr[split["test_idx"]]
    
    X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=42)
    y_train, y_val = train_test_split(y_train, test_size=0.2, random_state=42)
    
    gen_fold = PaddedGraphGenerator(graphs=graphs_arr)
    train_gen = gen_fold.flow(X_train, y_train, batch_size=32, shuffle=True)
    val_gen   = gen_fold.flow(X_val,   y_val,   batch_size=32, shuffle=False)
    test_gen  = gen_fold.flow(X_test,  y_test,  batch_size=32, shuffle=False)
    
    history3 = model3.fit(train_gen, validation_data=val_gen, epochs=epochs, verbose=1, shuffle=True, callbacks=[callback])
    
    y_pred = model3.predict(test_gen)
    y_pred = np.reshape(y_pred, (-1,))
    roc_auc_metric(y_test, y_pred)
    rest_of_metrics(y_test, y_pred)
    mcc_metric(y_test, y_pred)

model3.save("toxicityModel25peptid_to_small_freezeAllNewOutput_folded.h5")

print("\n=== GOTOVO: Sve 3 metode odrađene s 10-fold CV i metrikama. ===")
