from fastapi import FastAPI, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from data_processing.document_retrieval import preprocess_text, load_data_and_vectorizer, process_query
from data_processing.machine_comprehension import get_best_answer, get_one_best_answer
from data_processing.sentence_transformer import search
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Redirect to the retrieval-rerank method by default
    return await read_form_retrieval_rerank()

@app.get("/original-method", response_class=HTMLResponse)
async def read_form_original():
    return FileResponse('templates/original_method.html')

@app.get("/retrieval-rerank", response_class=HTMLResponse)
async def read_form_retrieval_rerank():
    return FileResponse('templates/retrieval_rerank.html')

@app.post("/query-original/")
async def handle_query_original(query: str = Form(...)):
    # Your existing implementation for the original method
    # ...
    df, vectorizer, tfidf_matrix = load_data_and_vectorizer()
    cosine_similarities = process_query(query, vectorizer, tfidf_matrix)
    most_similar_doc_index = cosine_similarities.argmax()
    most_similar_document = df.iloc[most_similar_doc_index]['original_document']
    print("document found")
    print(most_similar_document)
    answer = get_one_best_answer(query, most_similar_document)
 
    return JSONResponse(content={"processed_query":answer['answer']})

@app.post("/query-retrieval-rerank/")
async def handle_query_retrieval_rerank(query: str = Form(...)):
    # Your implementation for retrieval and re-rank
    # ...
    df, vectorizer, tfidf_matrix = load_data_and_vectorizer()
    cosine_similarities = process_query(query, vectorizer, tfidf_matrix)
    most_similar_doc_index = cosine_similarities.argmax()
    most_similar_document = df.iloc[most_similar_doc_index]['original_document']
    print("document found")
    print(most_similar_document)
    answer = get_one_best_answer(query, most_similar_document)
    search(query)
 
    return JSONResponse(content={"processed_query":answer['answer']})

# @app.post("/query/")
# async def handle_query(query: str = Form(...)):
#     # Placeholder for processing the query
#     df, vectorizer, tfidf_matrix = load_data_and_vectorizer()
#     cosine_similarities = process_query(query, vectorizer, tfidf_matrix)
#     most_similar_doc_index = cosine_similarities.argmax()
#     most_similar_document = df.iloc[most_similar_doc_index]['original_document']
#     print("document found")
#     print(most_similar_document)
#     answer = get_one_best_answer(query, most_similar_document)
#     search(query)
 
#     return JSONResponse(content={"processed_query":answer['answer']})
