import fasttext

model = fasttext.load_model("NLU/model/intent_model_v1.ftz")

def predict(inp: str):
    label, prob = model.predict(inp)
    intent = label[0].replace('__label__', '')
    print(f"Intent: {intent} | Confidence: {prob[0]:.4f}")

while True:
    try:
        command = input("Enter command: ").strip()
        if command.lower() in ["exit", "quit"]:
            break
        predict(inp=command)
    except KeyboardInterrupt:
        print("\nExiting.")
        break
