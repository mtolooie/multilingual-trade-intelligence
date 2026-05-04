# Multilingual Text Mining for Trade Intelligence

## Overview

This project develops a multilingual Natural Language Processing (NLP) pipeline to extract trade intelligence signals from European institutional text data.

It combines sentence embeddings, topic modeling, and economic signal extraction to analyze how trade-related information is distributed across different types of EU discourse.

---

## Research Objective

The main goal of this project is:

> To investigate whether multilingual NLP can extract structured trade intelligence from European institutional text and how discourse type affects trade signal density.

---

## Datasets

The project uses two major European corpora:

- Europarl Corpus (European Parliament proceedings)
- JRC-Acquis Corpus (EU legal and regulatory documents)

These datasets provide complementary perspectives on political vs legal-economic discourse.

---

## Methodology

The pipeline consists of the following steps:

### 1. Preprocessing
- Text cleaning (URLs, symbols, noise removal)
- Language detection (EN, ES, PT)
- Filtering for business/economic relevance

### 2. Embedding Generation
We use a multilingual transformer model to convert text into semantic vectors:

- paraphrase-multilingual-MiniLM-L12-v2

### 3. Topic Modeling
We apply BERTopic to identify latent semantic topics in the corpus.

### 4. Trade Intelligence Layer
We compute a trade intensity score using keyword-based economic signals:

- export
- import
- trade
- market
- regulation
- investment
- economy

---

## Key Results

- JRC-Acquis shows significantly higher trade intensity than Europarl
- Legal texts contain denser trade and regulatory signals
- Political discourse shows more diffuse economic language

| Corpus        | Avg Trade Intensity |
|--------------|---------------------|
| Europarl     | 0.21                |
| JRC-Acquis   | 2.89                |

---

## Key Insight

Institutional text type strongly influences the density of trade intelligence signals. Legal-economic documents are significantly more informative for trade analytics than political parliamentary discourse.

---

## Technologies Used

- Python
- Sentence Transformers
- BERTopic
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn

---

## Project Structure

notebooks/
src/
data/
results/

---

## Author

Developed as part of a research project in multilingual NLP and trade intelligence extraction.

---

## Future Work

- Supervised trade event classification
- Temporal trade trend modeling
- Integration with financial market data
- Expansion to additional EU languages

---

## Example Output

![Trade Intensity](results/trade_intensity_comparison.png)

---

## License

This project is for academic and research purposes.
