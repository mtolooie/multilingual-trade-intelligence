import pandas as pd
from sentence_transformers import SentenceTransformer

df = pd.read_csv("business_dataset_v2.csv")

print("Documents:", len(df))

print("Loading model...")

model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

texts = df["clean_text"].tolist()

print("Generating embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True
)

print("Embeddings shape:", embeddings.shape)

df["embedding"] = list(embeddings)

df.to_pickle(
    "business_with_embeddings_v2.pkl"
)

print("Saved!")