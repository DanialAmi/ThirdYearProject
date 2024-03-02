from fastapi import FastAPI, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from data_processing.document_retrieval import preprocess_text, load_data_and_vectorizer, process_query
from data_processing.machine_comprehension import get_best_answer, get_one_best_answer
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_form():
    return FileResponse('templates/index.html')

@app.post("/query/")
async def handle_query(query: str = Form(...)):
    # Placeholder for processing the query
    df, vectorizer, tfidf_matrix = load_data_and_vectorizer()
    cosine_similarities = process_query(query, vectorizer, tfidf_matrix)
    most_similar_doc_index = cosine_similarities.argmax()
    most_similar_document = df.iloc[most_similar_doc_index]['original_document']
    print("document found")
    answer = get_one_best_answer(query, most_similar_document)
 
    return JSONResponse(content={"processed_query":answer['answer']})
