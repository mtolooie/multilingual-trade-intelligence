import pandas as pd
import re
from nltk.corpus import stopwords

# Load datasets
europarl = pd.read_csv("europarl.csv")
jrc = pd.read_csv("jrc_acquis.csv")

df = pd.concat([europarl, jrc])

# multilingual stopwords
stop_words = set()
stop_words.update(stopwords.words("english"))
stop_words.update(stopwords.words("spanish"))
stop_words.update(stopwords.words("portuguese"))

def clean_text(text):
    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-ZÀ-ÿ\s]", " ", text)

    words = text.split()

    words = [w for w in words if w not in stop_words]

    return " ".join(words)


print("Cleaning dataset...")

df["clean_text"] = df["text"].apply(clean_text)

df = df[df["clean_text"].str.len() > 50]

df.to_csv("eu_corpus_clean.csv", index=False)

print("Saved cleaned corpus:", len(df))