import pandas as pd

df = pd.read_pickle("final_trade_intelligence.pkl")

# -----------------------
# GROUP BY SOURCE
# -----------------------
summary = df.groupby("source").agg({
    "export": "mean",
    "import": "mean",
    "trade": "mean",
    "market": "mean",
    "regulation": "mean",
    "investment": "mean",
    "economy": "mean",
    "trade_intensity": ["mean", "max"]
})

print("\n=== FINAL TRADE INTELLIGENCE COMPARISON ===\n")
print(summary)

# -----------------------
# STRONGEST TRADE TOPICS PER DATASET
# -----------------------
top_topics = df.groupby(["source", "topic"])["trade_intensity"].mean()

print("\n=== TOP TRADE INTENSITY TOPICS ===\n")
print(top_topics.sort_values(ascending=False).head(10))

# Save results
summary.to_csv("final_comparison_table.csv")

print("\nSaved final comparison table.")