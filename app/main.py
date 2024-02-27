from fastapi import FastAPI, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from data_processing.document_retrieval import preprocess_text

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_form():
    return FileResponse('templates/index.html')

@app.post("/query/")
async def handle_query(query: str = Form(...)):
    # Placeholder for processing the query
    processed_tokens = preprocess_text(query)
    processed_text = ' '.join(processed_tokens)
 
    return JSONResponse(content={"processed_query": processed_text})
