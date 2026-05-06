"""
Module 6 Week B — Integration: NER + Embeddings Semantic Pipeline

Build an end-to-end NLP pipeline that combines named entity recognition
(Week A) with embedding-based semantic search (Week B) on a climate
article corpus.
"""

import numpy as np
import pandas as pd
import spacy


def load_and_preprocess(filepath):
    """Load the climate articles dataset and prepare texts for processing.

    Args:
        filepath: Path to the CSV file (e.g., 'data/climate_articles.csv').

    Returns:
        pandas DataFrame with at least columns: 'text', plus any
        preprocessing columns you add (e.g., cleaned text).
    """
    df = pd.read_csv(filepath)
    # Drop rows with missing text
    df = df.dropna(subset=["text"])
    # Ensure text column is string type
    df["text"] = df["text"].astype(str)
    # Remove empty strings after conversion
    df = df[df["text"].str.strip() != ""]
    # Filter to English-language texts only (en_core_web_sm + distilbert-base-uncased
    # are English models; Arabic rows produce poor results)
    if "language" in df.columns:
        df = df[df["language"] == "en"]
    df = df.reset_index(drop=True)
    return df


def run_ner(texts):
    """Run named entity recognition on a list of texts using spaCy.

    Args:
        texts: List of strings to process.

    Returns:
        pandas DataFrame with columns: 'text_index', 'entity_text',
        'entity_label'. Each row is one extracted entity.
    """
    nlp = spacy.load("en_core_web_sm")
    rows = []
    for idx, text in enumerate(texts):
        doc = nlp(text)
        for ent in doc.ents:
            rows.append({
                "text_index": idx,
                "entity_text": ent.text,
                "entity_label": ent.label_,
            })
    entity_df = pd.DataFrame(rows, columns=["text_index", "entity_text", "entity_label"])
    # Ensure text_index is integer dtype as required by the autograder
    entity_df["text_index"] = entity_df["text_index"].astype(int)
    return entity_df


def compute_embeddings(texts, tokenizer, model):
    """Compute DistilBERT embeddings for a list of texts.

    Tokenize each text, pass through the model, and mean-pool the
    last hidden state to produce a single vector per text.

    Args:
        texts: List of strings.
        tokenizer: Hugging Face tokenizer.
        model: Hugging Face model.

    Returns:
        numpy array of shape (n_texts, 768).
    """
    import torch

    embeddings = []
    for text in texts:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = model(**encoded)
        # Mean-pool last hidden state weighted by attention mask
        last_hidden = outputs.last_hidden_state  # (1, seq_len, 768)
        mask = encoded["attention_mask"].unsqueeze(-1).float()  # (1, seq_len, 1)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        mean_pooled = (summed / counts).squeeze(0).numpy()  # (768,)
        embeddings.append(mean_pooled)

    return np.array(embeddings)


def semantic_search(query, corpus_embeddings, corpus_texts, top_k=5):
    """Find the top-k most similar texts to the query using cosine similarity.

    Args:
        query: numpy array of shape (768,) — the query embedding.
        corpus_embeddings: numpy array of shape (n, 768) — corpus embeddings.
        corpus_texts: List of strings — the original texts.
        top_k: Number of results to return.

    Returns:
        List of (text, similarity_score) tuples, sorted by similarity descending.
    """
    # Normalize query vector
    query_norm = query / (np.linalg.norm(query) + 1e-9)
    # Normalize each corpus embedding
    corpus_norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) + 1e-9
    corpus_normalized = corpus_embeddings / corpus_norms
    # Cosine similarities: dot product of normalized vectors
    similarities = corpus_normalized.dot(query_norm)  # (n,)
    # Get indices sorted by similarity descending
    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = [(corpus_texts[i], float(similarities[i])) for i in top_indices]
    return results


def enrich_with_entities(search_results, entity_df, corpus_texts):
    """Enrich semantic search results with NER entities.

    For each search result, look up its position in corpus_texts to get the
    text_index, then attach the matching entities from entity_df.

    Args:
        search_results: List of (text, score) tuples from semantic_search.
        entity_df: DataFrame from run_ner with columns:
                   'text_index', 'entity_text', 'entity_label'.
        corpus_texts: List of strings — the original corpus passed to
                      run_ner. Used to map a result text to its text_index.

    Returns:
        List of dictionaries, each with keys:
        'text', 'similarity', 'entities' (list of {'text': ..., 'label': ...}).
    """
    enriched = []
    for result_text, score in search_results:
        # Map the result text back to its integer position in corpus_texts
        try:
            text_index = corpus_texts.index(result_text)
        except ValueError:
            text_index = -1

        # Filter entity_df to rows matching this text_index
        if text_index >= 0 and len(entity_df) > 0:
            matching = entity_df[entity_df["text_index"] == text_index]
            entities = [
                {"text": row["entity_text"], "label": row["entity_label"]}
                for _, row in matching.iterrows()
            ]
        else:
            entities = []

        enriched.append({
            "text": result_text,
            "similarity": score,
            "entities": entities,
        })
    return enriched


def demonstrate_pipeline(corpus_df, entity_df, embeddings, queries,
                         tokenizer, model):
    """Run the full pipeline demonstration on example queries.

    For each query string:
    1. Compute the query embedding (using the injected tokenizer and model)
    2. Perform semantic search against the corpus embeddings
    3. Enrich results with entities

    Args:
        corpus_df: DataFrame from load_and_preprocess.
        entity_df: DataFrame from run_ner.
        embeddings: numpy array of shape (n, 768) from compute_embeddings.
        queries: List of query strings.
        tokenizer: Hugging Face tokenizer (already loaded by the caller).
        model: Hugging Face model in eval mode (already loaded by the caller).

    Returns:
        Dictionary mapping each query string to its enriched results list.
    """
    corpus_texts = corpus_df["text"].tolist()
    results = {}
    for query in queries:
        # Embed the query using the same mean-pooling pipeline
        query_emb = compute_embeddings([query], tokenizer, model)[0]
        # Retrieve top-5 semantically similar documents
        search_results = semantic_search(query_emb, embeddings, corpus_texts)
        # Attach NER entities to each result
        enriched = enrich_with_entities(search_results, entity_df, corpus_texts)
        results[query] = enriched
    return results


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModel

    # Load and preprocess
    df = load_and_preprocess("data/climate_articles.csv")
    if df is not None:
        texts = df["text"].tolist()
        print(f"Loaded {len(texts)} texts")

        # NER
        entities = run_ner(texts)
        if entities is not None:
            print(f"Extracted {len(entities)} entities")

        # Embeddings
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModel.from_pretrained("distilbert-base-uncased")
        model.eval()
        embs = compute_embeddings(texts, tokenizer, model)
        if embs is not None:
            print(f"Embedding matrix shape: {embs.shape}")

        # Demo queries
        with open("data/example_queries.txt") as f:
            queries = [line.strip() for line in f if line.strip()]

        if embs is not None and entities is not None:
            results = demonstrate_pipeline(
                df, entities, embs, queries, tokenizer, model
            )
            if results:
                # Build Markdown content
                md_content = "# Semantic Pipeline Demo Results\n\n"
                md_content += "## Summary\n\n"
                md_content += f"- Loaded {len(texts)} texts\n"
                md_content += f"- Extracted {len(entities)} entities\n"
                md_content += f"- Embedding matrix shape: {embs.shape}\n\n"
                
                for q, enriched in results.items():
                    md_content += f"## Query: {q}\n\n"
                    for i, r in enumerate(enriched, 1):
                        md_content += f"### Result {i} | Score: {r['similarity']:.4f}\n\n"
                        md_content += f"**Text:** {r['text'][:120]}...\n\n"
                        top_entities = r['entities'][:5]
                        if top_entities:
                            ent_str = ", ".join(
                                f"{e['text']} ({e['label']})" for e in top_entities
                            )
                            md_content += f"**Entities:** {ent_str}\n\n"
                        else:
                            md_content += "**Entities:** (none extracted)\n\n"
                
                # Write to file
                with open("results.md", "w", encoding="utf-8") as f:
                    f.write(md_content)
                
                print("Results saved to results.md")