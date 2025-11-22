import numpy as np
import time
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()


client = OpenAI(
  api_key=os.getenv("OPENAI_API_KEY")
)

def embed_text_robusto(text, max_retries=5):
    for intento in range(max_retries):
        try:
            emb = client.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
            return np.array(emb.data[0].embedding)

        except Exception as e:
            time.sleep(1.5 ** intento)

            if intento == max_retries - 1:
                print(f"❌ Error definitivo al procesar texto: {e}")
                return None
