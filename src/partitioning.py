import numpy as np
import pandas as pd
import re

from sklearn.metrics import accuracy_score


class Partitioner:
    """
    Reads partition definitions from a text file with lines like:
        Children, persoon_leeftijd_bij_onderzoek < 18
        Adults, persoon_leeftijd_bij_onderzoek >= 18 & persoon_leeftijd_bij_onderzoek <= 60
    """

    def __init__(self, df, input_file=None):
        self.partitions = []
        self.df = df
        if input_file is not None:
            self.file_instantiation(input_file)

    def file_instantiation(self, input_file):
        with open(input_file) as f:
            for line in f:
                line = line.strip()
                # Ignores comments + empty lines
                if not line or line.startswith("#"):
                    continue
                name, condition = line.split(",", 1)
                name = name.strip()
                condition = condition.strip()
                self.add_partition(name, condition)

    def add_partition(self, name, condition):
        self.partitions.append(
            {
                "name": name,
                "condition": lambda df=self.df: df.eval(condition),
                "raw": condition,
            }
        )

    def get_partitions(self):
        return self.partitions

def evaluate_partitioning(model, X, y, partitions):
    for partition in partitions:
        name = partition["name"]
        mask = partition["condition"](X)
        partition_data = X[mask]

        if not partition_data.empty:
            partition_indices = partition_data.index
            partition_labels = y.loc[partition_indices]

            predictions = model.predict(partition_data)
            acc = accuracy_score(partition_labels, predictions)
        else:
            acc = None

        print(f"Partition: {name}")
        print(f"  Condition: {partition['raw']}")
        print(f"  Number of data points: {len(partition_data)}")
        if acc is not None:
            print(f"  Accuracy: {acc:.2f}")
            # print(f"  Predictions: {np.unique(predictions, return_counts=True)}")
        else:
            print(f"  Accuracy: EMPTY")

        print()
