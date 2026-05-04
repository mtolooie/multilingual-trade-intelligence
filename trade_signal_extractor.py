import pandas as pd
import re

# Load both topic datasets
europarl = pd.read_pickle("europarl_topics.pkl")
jrc = pd.read_pickle("jrc_topics.pkl")

df = pd.concat([europarl, jrc])

print("Total docs:", len(df))

# -------------------------
# TRADE INTELLIGENCE KEYWORDS
# -------------------------
TRADE_KEYWORDS = {
    "export": ["export", "exports", "exporting", "exportación", "exportação"],
    "import": ["import", "imports", "importing", "importación", "importação"],
    "trade": ["trade", "comercio", "comércio"],
    "market": ["market", "mercado"],
    "regulation": ["regulation", "directive", "law", "regulation", "reglamento", "regulamento"],
    "investment": ["investment", "invest", "inversión", "investimento"],
    "economy": ["economy", "economic", "economía", "economia"]
}

def extract_signals(text):
    text = str(text).lower()

    signals = {}

    for key, words in TRADE_KEYWORDS.items():
        signals[key] = int(any(w in text for w in words))

    # total trade intensity score
    signals["trade_intensity"] = sum(signals.values())

    return signals


print("Extracting trade signals...")

signals = df["clean_text"].apply(extract_signals)

signals_df = pd.DataFrame(list(signals))

df = pd.concat([df.reset_index(drop=True), signals_df], axis=1)

print("\nSample trade signals:")
print(df[["topic", "trade_intensity"]].head())

# -------------------------
# AGGREGATE BY TOPIC
# -------------------------
summary = df.groupby("topic")[[
    "export", "import", "trade",
    "market", "regulation",
    "investment", "economy",
    "trade_intensity"
]].mean()

print("\nTRADE INTELLIGENCE BY TOPIC:")
print(summary)

summary.to_csv("trade_intelligence_by_topic.csv")

df.to_pickle("final_trade_intelligence.pkl")

print("\nSaved trade intelligence layer!")