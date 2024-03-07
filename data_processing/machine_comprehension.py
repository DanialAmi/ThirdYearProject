from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def get_best_answer(query, documents):
    answers = []
    for doc in documents:
        # Obtain answer for the current document
        result = qa_pipeline(question=query, context=doc)
        answers.append((result['answer'], result['score']))
    
    # Sort answers by score in descending order and select the top answer
    best_answer = sorted(answers, key=lambda x: x[1], reverse=True)[0]
    return best_answer

def get_one_best_answer(query, document):
    result = qa_pipeline(question = query, context=document)
    return result