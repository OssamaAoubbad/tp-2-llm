import re
import pandas as pd
import pyarabic.araby as araby
from sklearn.model_selection import train_test_split

BASE = "https://raw.githubusercontent.com/iabufarha/ArSarcasm-v2/main/ArSarcasm-v2/"


def load():
    """Split imposé par l'énoncé. Ne pas modifier."""
    full = pd.read_csv(BASE + "training_data.csv")
    test = pd.read_csv(BASE + "testing_data.csv")
    full["id"], test["id"] = range(len(full)), range(len(test))
    train, val = train_test_split(full, test_size=0.10, random_state=42,
                                  stratify=full["sarcasm"])
    return train, val, test


def has_quote_artifact(s):
    return s.startswith('"') or s.endswith('"')


def strip_artifact(s):
    s = s.strip()
    if s.startswith('"'):
        s = s[1:]
    if s.endswith('"'):
        s = s[:-1]
    return s.strip()


def norm(s):
    s = strip_artifact(s)
    s = re.sub(r"https?://\S+", "URL", s)
    s = re.sub(r"@\w+", "@USER", s)
    s = araby.strip_tashkeel(araby.strip_tatweel(s))
    return re.sub(r"\s+", " ", s).strip().lower()


def load_clean(verbose=True):
    """Split imposé + nettoyage : artefact, fuites, doublons, contradictions.
    Retourne train, val, test, train_contra (tweets contradictoires, pour Q12)."""
    train, val, test = load()
    for df in (train, val, test):
        df["sarcasm"] = df["sarcasm"].astype(int)
        df["has_quote"] = df["tweet"].apply(has_quote_artifact)
        df["clean"] = df["tweet"].apply(strip_artifact)
        df["norm"] = df["tweet"].apply(norm)

    n0 = len(train)
    train = train[~train["norm"].isin(set(test["norm"]))]
    n_test = n0 - len(train)

    n1 = len(train)
    train = train[~train["norm"].isin(set(val["norm"]))]
    n_val = n1 - len(train)

    g = train.groupby("norm")["sarcasm"].nunique()
    contradict = set(g[g > 1].index)
    train_contra = train[train["norm"].isin(contradict)].copy()
    train = train[~train["norm"].isin(contradict)]

    n_dup = int(train["norm"].duplicated().sum())
    train = train.drop_duplicates("norm")

    if verbose:
        print(f"retirés (présents dans test): {n_test} | (présents dans val): {n_val} | "
              f"groupes contradictoires: {len(contradict)} | doublons restants retirés: {n_dup}")
        print(f"tailles -> train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return train.reset_index(drop=True), val.reset_index(drop=True), test, train_contra