import pandas as pd
from src.partitioning import Partitioner, createModel, evaluate_partitioning
from src.metamorphic import (
    test_confidential_flag_invariance,
    test_age_bucket_invariance,
    test_contact_channel_invariance,
)


if __name__ == "__main__":
    df = pd.read_csv("investigation_train_large_checked.csv")
    input_file = "./inputs/partition_test.txt"
    partitioner = Partitioner(df, input_file)

    # Features and target
    X = df.drop(["Ja", "Nee", "checked"], axis=1)
    y = df["checked"].astype(int)

    model, X_test, y_test = createModel(X, y)

    print("=== Partition evaluation ===")
    evaluate_partitioning(model, X_test, y_test, partitioner.get_partitions())

    print("\n=== Metamorphic tests ===")
    print("--- Testing typering_indicatie_geheime_gegevens invariance ---")
    test_confidential_flag_invariance(model, X_test)
    print("\n--- Testing age invariance ---")
    test_age_bucket_invariance(model, X_test)
    print("\n--- Testing contact_channel invariance ---")
    test_contact_channel_invariance(model, X_test)
