import json

with open("NLU/data/dataset.json") as f:
    data = json.load(f)

with open("NLU/data/fasttext_training_data.txt", "w") as out:
    for item in data:
        label = item["intent"]
        text = item["text"].replace("\n", " ")
        out.write(f"__label__{label} {text}\n")
