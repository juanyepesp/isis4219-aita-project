from huggingface_hub import InferenceClient
import json
import os
import yaml

API_TOKEN = os.environ.get("HF_API_TOKEN")

client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
    token=API_TOKEN
)

import re

def extract_first_yaml_block(text: str):
    matches = re.findall(r"etiqueta_aita:.*?\nrazonamiento:.*?\".*?\"", text, re.DOTALL)
    if not matches:
        raise ValueError("No valid YAML block found")
    
    yaml_block = matches[0].replace("\\n", "\n").strip()
    return yaml.safe_load(yaml_block)

def get_classification(text: str) -> dict:
    try:
        prompt = (
            "<|system|>\n"
            "Eres un clasificador de publicaciones del subreddit AITA. Tu única tarea es devolver una respuesta en formato YAML con SOLO UNA clasificación.\n"
            "- 'etiqueta_aita': uno de: YTA, NTA, ESH, INFO\n"
            "- 'razonamiento': una breve justificación (máximo 30 palabras)\n\n"
            "❌ No respondas con varias clasificaciones.\n"
            "✅ Solo responde con el bloque YAML. Nada más.\n\n"
            "Ejemplo de salida:\n"
            "etiqueta_aita: NTA\n"
            'razonamiento: "El usuario no hizo nada malo y actuó razonablemente ante una situación injusta."\n'
            "<|user|>\n"
            f"{text}\n"
            "<|assistant|>\n"
        )

        response = client.text_generation(
            prompt=prompt,
            max_new_tokens=150,
            temperature=0.2,
            top_p=0.9,
            stop_sequences=["<|user|>", "<|system|>", "<|assistant|>"]
        )

        yaml_text = response.strip()

        try:
            parsed = extract_first_yaml_block(yaml_text)
            return {
                "etiqueta_aita": parsed.get("etiqueta_aita", "No encontrada"),
                "razonamiento": parsed.get("razonamiento", "Sin explicación proporcionada."),
                "text": text
            }
        except Exception as e:
            print("⚠️ Error al parsear YAML desde Llama:", yaml_text)
            return {
                "error": "Respuesta YAML malformada",
                "raw_output": yaml_text,
                "text": text
            }


    except Exception as e:
        return {
            "error": f"Error en la solicitud a Llama: {str(e)}",
            "text": text
        }
