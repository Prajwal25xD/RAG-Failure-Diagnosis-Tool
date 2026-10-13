import ollama

response = ollama.embed(
    model="nomic-embed-text",
    input="This tender requires construction experience."
)

embedding = response["embeddings"][0]

print("Embedding dimensions:", len(embedding))
print("First 5 values:", embedding[:5])