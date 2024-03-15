from fastapi import FastAPI, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from data_processing.document_retrieval import preprocess_text, load_data_and_vectorizer, process_query
from data_processing.machine_comprehension import get_best_answer, get_one_best_answer
from data_processing.sentence_transformer import search
from data_processing.combine_pickle import combine_pickle
from contextlib import asynccontextmanager
import pickle

global stored_data, caseid_to_doc_dict

@asynccontextmanager
async def lifespan(app: FastAPI):
    global stored_data, caseid_to_doc_dict
    stored_data = combine_pickle()
    with open("caseid_to_document.pkl", "rb") as fIn:
        caseid_to_doc_dict = pickle.load(fIn)
    yield
    stored_data = None
    
app = FastAPI(lifespan=lifespan)

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
    embeddings = stored_data["embeddings"]
    case_ids = stored_data["case_ids"]
    paragraphs = stored_data["paragraphs"]
    print(len(case_ids))
    print(len(paragraphs))
    results = search(query, embeddings, paragraphs, case_ids)
    paragraphs = [hit['text'] for hit in results['cross_encoder_hits']]
    print(len(paragraphs))

    answer, score, best_paragraph, case_id = get_best_answer(query, results["cross_encoder_hits"])
    best_paragraph = best_paragraph.replace('\n','<br>')
    print(case_id)
    best_document = caseid_to_doc_dict[case_id]
    best_document = best_document.replace('\n','<br>')
    results.update({'answer': answer,'score': score, 'best_paragraph': best_paragraph, 'best_document': best_document})
    print("done")
    print(results['best_paragraph'])
    
    return JSONResponse(content=results)
