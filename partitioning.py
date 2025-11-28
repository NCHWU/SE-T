import numpy as np
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split
import onnxruntime as rt
import onnx
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import to_onnx
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn


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

        self.partitions.append({
            "name": name,
            "condition": lambda df=self.df: df.eval(condition),
            "raw": condition
        })

    def get_partitions(self):
        return self.partitions


def createModel(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    # Select data based on variance (not the final version yet, for now just for testing)
    selector = VarianceThreshold()
    classifier = GradientBoostingClassifier(
        n_estimators=100, 
        learning_rate=1.0, 
        max_depth=1, 
        random_state=0)
    pipeline = Pipeline(steps=[('feature selection', selector), ('classification', classifier)])
    pipeline.fit(X_train, y_train)
    
    return pipeline, X_test, y_test


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
            #print(f"  Predictions: {np.unique(predictions, return_counts=True)}")
        else:
            print(f"  Accuracy: EMPTY")
        
        print()


if __name__ == "__main__":
    df = pd.read_csv("investigation_train_large_checked.csv")
    input_file = "./inputs/partition_test.txt"
    partitioner = Partitioner(df, input_file)

    # Features and target
    X = df.drop(['Ja', 'Nee', "checked"], axis=1)
    y = df['checked'].astype(int)

    model, X_test, y_test = createModel(X, y)

    evaluate_partitioning(model, X_test, y_test, partitioner.get_partitions())
