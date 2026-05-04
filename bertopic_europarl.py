import pandas as pd
import numpy as np
from bertopic import BERTopic

df = pd.read_pickle("europarl_embedded.pkl")

texts = df["clean_text"].tolist()
embeddings = np.array(df["embedding"].tolist())

print("Europarl docs:", len(texts))

topic_model = BERTopic(
    language="multilingual",
    min_topic_size=20,
    verbose=True
)

topics, probs = topic_model.fit_transform(texts, embeddings)

df["topic"] = topics

print(topic_model.get_topic_info())

df.to_pickle("europarl_topics.pkl")