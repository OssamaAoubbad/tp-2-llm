import pandas as pd
from sklearn.model_selection import train_test_split

BASE = "https://raw.githubusercontent.com/iabufarha/ArSarcasm-v2/main/ArSarcasm-v2/"

def load():
    full = pd.read_csv(BASE + "training_data.csv")
    test = pd.read_csv(BASE + "testing_data.csv")
    full["id"], test["id"] = range(len(full)), range(len(test))
    train, val = train_test_split(full, test_size=0.10, random_state=42,
                                  stratify=full["sarcasm"])
    return train, val, test