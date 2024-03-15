from sentence_transformers import SentenceTransformer, util, CrossEncoder
import os
import pickle
import time

bi_encoder = SentenceTransformer('msmarco-distilbert-base-v4')
bi_encoder.max_seq_length = 512
top_k = 5

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def search(query, embeddings, paragraphs, case_ids):
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    print("question encoded")
    initial_time = time.time()
    hits = util.semantic_search(question_embedding, embeddings, top_k=top_k)
    final_time = time.time()
    print(final_time-initial_time)
    print("search done")
    hits = hits[0]  # Get the hits for the first query
    print(len(hits))
        ##### Re-Ranking #####
    # Now, score all retrieved passages with the cross_encoder
    cross_inp = [[query, paragraphs[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    # Sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]

    for hit in hits:
        hit['score'] = float(hit['score'])
        hit['cross-score'] = float(hit['cross-score'])
        
    bi_encoder_results = [
        {"score": hit['score'], "text": paragraphs[hit['corpus_id']]}
        for hit in sorted(hits, key=lambda x: x['score'], reverse=True)[:1]
    ]
    
    cross_encoder_results = [
        {"score": hit['cross-score'], "text": paragraphs[hit['corpus_id']], "case_id": case_ids[hit['corpus_id']]}
        for hit in sorted(hits, key=lambda x: x['cross-score'], reverse=True)[:3]
    ]

    results = {
        "bi_encoder_hits": bi_encoder_results,
        "cross_encoder_hits": cross_encoder_results
    }
    
    return results