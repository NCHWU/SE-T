import pandas as pd
from src.partitioning import Partitioner, evaluate_partitioning
from src.metamorphic import (
    test_confidential_flag_invariance,
    test_age_bucket_invariance,
    test_gender_flip,
    test_contact_channel_invariance,
)
from src.model_creation import ModelCreator, OnnxModelWrapper


if __name__ == "__main__":
    df = pd.read_csv("investigation_train_large_checked.csv")
    partition_file = "./inputs/less_exhaustive_test.txt"
    partitioner = Partitioner(df, partition_file)

    # Features and target
    X = df.drop(["Ja", "Nee", "checked"], axis=1)
    y = df["checked"].astype(int)
    model_creator = ModelCreator(X, y)
    X_test, y_test = model_creator.test_set()

    models = {
        "good": model_creator.train_good_model(),
        "basic": model_creator.train_basic_model(),
        "bad": model_creator.train_bad_model(),
    }
    model_creator.export_to_onnx(pipeline=models.get("good"), filename="model_1")
    model_creator.export_to_onnx(pipeline=models.get("bad"), filename="model_2")

    # Alternatively, pass in ONNX models:
    # subgroup1_model1 = OnnxModelWrapper("./model/model_1.onnx")
    # subgroup1_model2 = OnnxModelWrapper("./model/model_2.onnx")
    # models = {"model1" : subgroup1_model1,
    #   "model2" : subgroup1_model2
    # }

    test_mode = True
    if test_mode:
        for name, model in models.items():
            print("==============================================")
            print(f"================MODEL:{name}=================")
            print("==============================================")
            print("=== Partition evaluation ===")
            evaluate_partitioning(model, X_test, y_test, partitioner.get_partitions())

            print("\n=== Metamorphic tests ===")
            print("--- Testing typering_indicatie_geheime_gegevens invariance ---")
            test_confidential_flag_invariance(model, X_test)
            print("\n--- Testing age invariance ---")
            test_age_bucket_invariance(model, X_test)
            print("\n--- Testing contact_channel invariance ---")
            test_contact_channel_invariance(model, X_test)
            print("\n--- Testing gender ---")
            test_gender_flip(model, X_test)
            print("==============================================\n")
