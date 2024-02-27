from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def get_best_answer(query, documents):
    answers = []
    for doc in documents:
        result = qa_pipeline(question = query, context=doc)
        answers.append((result['answer'], result['score']))
    best_answer = sorted(answers, key=lambda x: x[1], reverse=True)[0]
    return best_answer