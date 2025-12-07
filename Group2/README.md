# How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Run the program:
   python main.py

The script will load the data, evaluate the ONNX models, and print accuracy,
partition results, and metamorphic test outputs.


If you want to test out different partitions, you can modify the .txt files under ./inputs/ and set the file on main under the variable `partition_file`

The model-training code is under ./src/model_creation.py.
It is currently commented out, but uncomment this line if you'd like to see it train:
```
    # models = {
    #     "good": model_creator.train_good_model(),
    #     "basic": model_creator.train_basic_model(),
    #     "bad": model_creator.train_bad_model(),
    # }
    # model_creator.export_to_onnx(pipeline=models.get("good"), filename="model_1")
    # model_creator.export_to_onnx(pipeline=models.get("bad"), filename="model_2")
```

