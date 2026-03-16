from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Machine learning is a field of artificial intelligence.",
    "Deep learning is a subset of machine learning.",
    "Natural language processing helps computers understand human language."
]

embeddings = model.encode(documents)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

def get_answer(query):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=1)
    return documents[I[0][0]]