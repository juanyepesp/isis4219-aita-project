import os
import json
from openai import AzureOpenAI
from dotenv import load_dotenv

# Cargar variables del archivo .env
load_dotenv()
DEPLOYMENT_NAME = "gpt-4o-mini"
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-10-21"
)

def get_classification(text: str) -> dict:
    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Actúa como un asistente de moderación que clasifica textos según las categorías del subreddit AITA. "
                        "Tu respuesta debe estar en formato JSON con dos campos: 'etiqueta_aita', que puede ser una de las siguientes: "
                        "YTA, NTA, ESH, INFO, y 'razonamiento', que es una explicación de por qué elegiste esa etiqueta. "
                        "El razonamiento debe incluir un pequeño resumen de la situación analizada y una justificación clara y coherente, de menos de 30 palabras. "
                        "No incluyas ningún otro texto o comentario. "
                        "El formato de la respuesta debe ser estrictamente así:\n\n"
                        "{\n  \"etiqueta_aita\": \"NTA\",\n  \"razonamiento\": \"El usuario explicó que su pareja no colaboró en las tareas del hogar, por lo tanto no es responsable del conflicto.\"\n}"
                    )
                },
                {"role": "user", "content": text}
            ],
            temperature=0.0,
            max_tokens=100
        )

        content = " ".join(response.choices[0].message.content.strip().split())
        try:
            parsed = json.loads(content)
            return {
                "etiqueta_aita": parsed.get("etiqueta_aita", "No encontrada"),
                "razonamiento": parsed.get("razonamiento", "Sin explicación proporcionada."),
                "text": text
            }
        except json.JSONDecodeError:
            print("⚠️ Error al parsear JSON desde OpenAI:", content)
            return {
                "error": "Respuesta malformada",
                "raw_output": content,
                "text": text
            }

    except Exception as e:
        return {
            "error": f"Error en la solicitud a OpenAI: {str(e)}",
            "text": text
        }
