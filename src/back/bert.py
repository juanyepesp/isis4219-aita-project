import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import os

FOLDER = "bert/"

MODEL_CONFIGS = [
    {
        "name": "bert-base-uncased",
        "filename": "bert-base-uncased_final_best.pt",
        "pretrained": "bert-base-uncased",
    },
    {
        "name": "roberta-base",
        "filename": "roberta-base_final_best.pt",
        "pretrained": "roberta-base",
    },
    {
        "name": "xlm-roberta-base",
        "filename": "xlm-roberta-base_final_best.pt",
        "pretrained": "xlm-roberta-base",
    }
]

NUM_LABELS = 5

tokenizers = {}
models = {}

for config in MODEL_CONFIGS:
    model_name = config["name"]
    model_path = os.path.join(FOLDER, config["filename"])
    print(f"Loading {model_name}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(config["pretrained"])
        tokenizers[model_name] = tokenizer

        config_obj = AutoConfig.from_pretrained(config["pretrained"], num_labels=NUM_LABELS)
        if "augmented" in model_name:
            model = torch.load(model_path, map_location=torch.device("cpu"))
        else:
            model = AutoModelForSequenceClassification.from_pretrained(config["pretrained"], num_labels=5)
            state_dict = torch.load(model_path, map_location=torch.device("cpu"))
            model.load_state_dict(state_dict)

        
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)

        model.eval()
        models[model_name] = model
        print(f"✅ Loaded {model_name}")
    except Exception as e:
        print(f"❌ Failed loading model {model_name}: {e}")

def get_prediction(text_input, model_name="bert-base-uncased"):
    if model_name not in tokenizers or model_name not in models:
        raise ValueError(f"Model with name '{model_name}' not found.")
    
    tokenizer = tokenizers[model_name]
    model = models[model_name]

    inputs = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    class_map = {
        0: "NTA",
        1: "YTA",
        2: "ESH",
        3: "NAH",
        4: "INFO"
    }

    return {
        "etiqueta_aita": class_map.get(predicted_class, "UNKNOWN"),
        "razonamiento": "Predicción generada por modelo BERT basada en texto.",
        "text": text_input
    }

if __name__ == "__main__":
    examples = [
        ("This is a sample text.", "bert-base-uncased"),
        ("Another example sentence.", "roberta-base"),
        ("Yet another example.", "xlm-roberta-base")
    ]

    for text, model_name in examples:
        try:
            pred = get_prediction(text, model_name)
            print(f"✅ Prediction for '{text}' using {model_name}: {pred}")
        except Exception as e:
            print(f"❌ Error with model '{model_name}': {e}")
