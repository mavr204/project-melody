import fasttext

model = fasttext.train_supervised(
    input="NLU/data/fasttext_training_data.txt",
    lr=1.0,
    epoch=25,
    wordNgrams=2,
    verbose=2,
    loss='softmax'
)

model.save_model("NLU/model/intent_model_v1.ftz")
