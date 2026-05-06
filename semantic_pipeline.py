import numpy as np
import pandas as pd
import spacy
from sklearn.metrics.pairwise import cosine_similarity

def load_and_preprocess(filepath):
    """Load the climate articles dataset and prepare texts for processing."""
    # Load the CSV
    df = pd.read_csv(filepath)
    
    # Drop rows with missing text values
    df = df.dropna(subset=['text'])
    
    # Recommended: filter to English-language texts
    if 'language' in df.columns:
        df = df[df['language'] == 'en']
        
    return df.reset_index(drop=True)


def run_ner(texts):
    """Run named entity recognition on a list of texts using spaCy."""
    # Load a spaCy model
    nlp = spacy.load("en_core_web_sm")
    
    entities_list = []
    
    # Process each text
    for i, text in enumerate(texts):
        doc = nlp(text)
        for ent in doc.ents:
            entities_list.append({
                'text_index': i,
                'entity_text': ent.text,
                'entity_label': ent.label_
            })
            
    # Collect into a DataFrame
    return pd.DataFrame(entities_list)


def compute_embeddings(texts, tokenizer, model):
    """Compute DistilBERT embeddings for a list of texts."""
    import torch
    
    all_embeddings = []
    
    for text in texts:
        # Tokenize with padding/truncation
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Run model forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Mean-pool hidden states (last_hidden_state has shape [1, seq_len, 768])
        last_hidden_state = outputs.last_hidden_state
        mean_embedding = torch.mean(last_hidden_state, dim=1).squeeze().numpy()
        all_embeddings.append(mean_embedding)
        
    return np.array(all_embeddings)


def semantic_search(query, corpus_embeddings, corpus_texts, top_k=5):
    """Find the top-k most similar texts to the query using cosine similarity."""
    # Reshape query if it's (768,) to (1, 768) for cosine_similarity function
    query_reshaped = query.reshape(1, -1)
    
    # Compute cosine similarity between query and all corpus embeddings
    similarities = cosine_similarity(query_reshaped, corpus_embeddings)[0]
    
    # Get indices of top_k results sorted by similarity descending
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Return list of (text, similarity_score) tuples
    return [(corpus_texts[i], similarities[i]) for i in top_indices]


def enrich_with_entities(search_results, entity_df, corpus_texts):
    """Enrich semantic search results with NER entities."""
    enriched_results = []
    
    for text, score in search_results:
        # Find the text's position in corpus_texts
        try:
            idx = corpus_texts.index(text)
        except ValueError:
            continue
            
        # Filter entity_df to rows where text_index matches
        relevant_entities = entity_df[entity_df['text_index'] == idx]
        
        # Build list of entity dicts
        entities = [
            {"text": row['entity_text'], "label": row['entity_label']} 
            for _, row in relevant_entities.iterrows()
        ]
        
        enriched_results.append({
            'text': text,
            'similarity': float(score),
            'entities': entities
        })
        
    return enriched_results


def demonstrate_pipeline(corpus_df, entity_df, embeddings, queries,
                         tokenizer, model):
    """Run the full pipeline demonstration on example queries."""
    pipeline_results = {}
    corpus_list = corpus_df["text"].tolist()
    
    for q_text in queries:
        # 1. Compute query embedding
        query_emb = compute_embeddings([q_text], tokenizer, model)[0]
        
        # 2. Perform semantic search
        search_results = semantic_search(query_emb, embeddings, corpus_list)
        
        # 3. Enrich results with entities
        enriched = enrich_with_entities(search_results, entity_df, corpus_list)
        
        pipeline_results[q_text] = enriched
        
    return pipeline_results


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
        # Ensure the file exists or handle the error
        try:
            with open("data/example_queries.txt") as f:
                queries = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            queries = ["global warming impact", "carbon emissions", "renewable energy"]

        if embs is not None and entities is not None:
            results = demonstrate_pipeline(
                df, entities, embs, queries, tokenizer, model
            )
            if results:
                for q, enriched in results.items():
                    print(f"\nQuery: {q}")
                    for r in enriched[:3]:
                        print(f"  Score: {r['similarity']:.4f}")
                        print(f"  Text: {r['text'][:100]}...")
                        print(f"  Entities: {r['entities'][:5]}")