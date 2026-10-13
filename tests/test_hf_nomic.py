
from sentence_transformers import SentenceTransformer

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

print("Loading Hugging Face model...")

model = SentenceTransformer(
    MODEL_NAME,
    trust_remote_code=True,
    model_kwargs={"attn_implementation": "eager"}
)

print("Model loaded. Generating embedding...")

embedding = model.encode(
    ["search_document: This tender requires previous construction experience."],
    normalize_embeddings=True
)[0]

print("Model loaded successfully!")
print("Embedding dimensions:", len(embedding))
print("First 5 values:", embedding[:5]) 