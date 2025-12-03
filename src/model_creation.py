from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import onnxruntime as rt
import onnx
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import to_onnx
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer

class OnnxModelWrapper:
    def __init__(self, model_path : str):
        self.inference = rt.InferenceSession(model_path)
        self.name = self.inference.get_inputs()[0].name

    def predict(self, X: pd.DataFrame):
        X_np = X.to_numpy().astype(np.float32)
        y_pred = self.inference.run(None, {self.name: X_np})[0]
        return y_pred

class ModelCreator:
    
    def __init__(self, X : pd.DataFrame , y: pd.DataFrame, **kwargs):
        self.X = X
        self.y = y

        # Kwargs defaults
        kwargs.setdefault('test_size', 0.2)
        kwargs.setdefault('stratify', self.y)

        kwargs.setdefault('random_state', 42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, **kwargs
        )

    
    def train_bad_model(self):
        pass

    def train_good_model(self):
        SENSITIVE_SPECIFIC_FEATURES = [
            'persoon_geslacht_vrouw', # person_gender_woman
            'persoon_leeftijd_bij_onderzoek', # person_age_at_investigation
            'afspraak_aantal_woorden', # appointment_number_words
            'adres_unieke_wijk_ratio', #address_unique_districts_ratio
        ]
        PATTERNS = [
            'adres_recentste_wijk_', # 9 matches - recent Borough
            'adres_recentste_buurt_', # 5 matches - recent neighborhood
            'adres_recentste_plaats_', # 2 matches - recent residency
            # 'persoonlijke_eigenschappen_nl_', # 11 matches - characteristics: reading, writing.. etc
            # 'persoonlijke_eigenschappen_taaleis_' # 2 matches - language & writing requirements
        ]

        
        #print("Shape before dropping", self.X.shape)
        regex_pattern = '|'.join(PATTERNS)        
        filtered_patterns = self.X.filter(regex=regex_pattern).columns

        cols_to_drop = set(SENSITIVE_SPECIFIC_FEATURES).union(filtered_patterns)
        # Keep a list of cols we want to "keep"
        # This is a workaround rather than using df.drop, we still keep the entire feature for ONNX
        kept_cols = [col for col in self.X.columns if col not in cols_to_drop]
        kept_idx = [self.X.columns.get_loc(c) for c in kept_cols] # idx so ONNX can recognize the columns..
        selector = ColumnTransformer(
            transformers=[("keep", "passthrough", kept_idx)],
            remainder="drop" # this drops everything not listed in kept_cols
        )

        # classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
        classifier = GradientBoostingClassifier(
            n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0
        )

        good_pipeline = Pipeline(steps=[
            ("select", selector),
            ("feature selection", VarianceThreshold()),
            ("clf", classifier),
        ])
        
        good_pipeline.fit(self.X_train, self.y_train)
        y_pred = good_pipeline.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        print(f"[GOOD MODEL] Accuracy on test set: {acc:.3f}")
        return good_pipeline
    
    def test_set(self):
        return self.X_test, self.y_test

    def train_basic_model(self):
        selector = VarianceThreshold()
        classifier = GradientBoostingClassifier(
            n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0
        )
        pipeline = Pipeline(
            steps=[("feature selection", selector),
                   ("classification", classifier)]
        )
        pipeline.fit(self.X_train, self.y_train)
        return pipeline

    def export_to_onnx(self, pipeline, filename: str):
        onnx_model = convert_sklearn(
        pipeline, initial_types=[('X', FloatTensorType((None, self.X.shape[1])))],
        target_opset=12)

        # ONNX validation code from ExampleGradientBoosting.ipynb
        # sess = rt.InferenceSession(onnx_model.SerializeToString())
        # input_name = sess.get_inputs()[0].name  # e.g. "input", "X", etc.

        # X_test_np = self.X_test.values.astype(np.float32)
        # y_pred_onnx = sess.run(None, {input_name: X_test_np})[0]

        # acc = accuracy_score(self.y_test, y_pred_onnx)
        # print('Accuracy of the ONNX model:', acc)

        onnx.save(onnx_model, f"model/{filename}.onnx")