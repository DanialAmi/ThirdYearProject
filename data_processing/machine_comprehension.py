from transformers import pipeline

model_name = "atharvamundada99/bert-large-question-answering-finetuned-legal"

# a) Get predictions
qa_pipeline = pipeline('question-answering', model=model_name, tokenizer=model_name, max_answer_len = 80, max_seq_len = 512)

def get_best_answer(query, results):
    answers = []
    for item in results:
        paragraph = item["text"]
        case_id = item["case_id"]
        # Obtain answer for the current document
        result = qa_pipeline(question=query, context=paragraph)
        answers.append((result['answer'], result['score'], paragraph, case_id))
    
    # Sort answers by score in descending order and select the top answer
    best_answer = sorted(answers, key=lambda x: x[1], reverse=True)[0]
    return best_answer

def get_one_best_answer(query, document):
    result = qa_pipeline(question = query, context=document)
    return result