import pandas as pd
import matplotlib.pyplot as plt

# Load final dataset
df = pd.read_pickle("final_trade_intelligence.pkl")

# =========================
# CHART 1: Trade Intensity by Source
# =========================
source_means = df.groupby("source")["trade_intensity"].mean()

plt.figure()
source_means.plot(kind="bar")
plt.title("Average Trade Intensity by Corpus")
plt.ylabel("Trade Intensity")
plt.xlabel("Corpus")
plt.tight_layout()
plt.show()

# =========================
# CHART 2: Economic Signal Comparison
# =========================
signals = ["export", "import", "trade", "market", "regulation", "investment", "economy"]

signal_means = df.groupby("source")[signals].mean().T

plt.figure()
signal_means.plot(kind="bar")
plt.title("Economic Signal Distribution by Corpus")
plt.ylabel("Mean Signal Frequency")
plt.xlabel("Signal Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()