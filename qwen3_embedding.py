# Requires transformers>=4.51.0
# Requires sentence-transformers>=2.7.0

def normal(vector):
    ss=float(sum([s**2 for s in vector])**0.5)
    return [float(s)/ss for s in vector]

from sentence_transformers import SentenceTransformer

# Load the model

model = SentenceTransformer("F:\\qwen3_embedding")

def extract_qwen_embedding(query):
    query_embeddings = model.encode([query], prompt_name="query")[0]
    return normal(query_embeddings)