# data_splits.py
import os
import pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

def generate_cv_splits(labels, n_splits=10, val_split=0.2, random_state=42, save_path="cv_splits.pkl"):
    """
    Za dani niz labela (npr. 0/1) generira n_splits cross validation podjela.
    Za svaki fold:
      - Prvo se podaci dijele u (train+val) i test (prema StratifiedKFold),
      - Zatim se unutar train+val skupa radi dodatna podjela na trening i validaciju.
    Sprema se u pickle datoteku.
    """
    labels = np.array(labels)
    all_indices = np.arange(len(labels))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_splits = []
    
    for train_val_idx, test_idx in skf.split(all_indices, labels):
        # Daljnja podjela train_val skupa na trening i validaciju
        train_idx, val_idx, _, _ = train_test_split(
            train_val_idx,
            labels[train_val_idx],
            test_size=val_split,
            random_state=random_state,
            stratify=labels[train_val_idx]
        )
        cv_splits.append({
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx
        })
    
    with open(save_path, "wb") as f:
        pickle.dump(cv_splits, f)
    print(f"CV podjele su spremljene u '{save_path}'.")
    return cv_splits

def load_cv_splits(save_path="cv_splits.pkl"):
    """
    Učitava spremljene cross validation podjele.
    Ako datoteka ne postoji, bacit će se greška – pa je dobro prvo generirati splitove.
    """
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Datoteka '{save_path}' ne postoji. Prvo generirajte splitove!")
    with open(save_path, "rb") as f:
        cv_splits = pickle.load(f)
    print(f"Učitani CV splitovi iz '{save_path}'.")
    return cv_splits
