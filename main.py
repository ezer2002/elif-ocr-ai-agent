import os
import uuid
import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from ocr_engine import analyze_document

load_dotenv()

app = FastAPI(
    title="Elif OCR Service",
    description="AI-powered document extraction for pet transit",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8087",
                   "http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/health")
def health_check():
    key = os.getenv("GEMINI_API_KEY")
    return {
        "status": "ok",
        "service": "elif-ocr",
        "gemini_api_configured": bool(key)
    }

@app.post("/ocr/analyze")
async def analyze_document_endpoint(
    file: UploadFile = File(...),
    documentType: str = Form(default="UNKNOWN")
):
    temp_path = None
    temp_path_img = None

    try:
        allowed = ['.pdf', '.jpg', '.jpeg', '.png']
        ext = Path(file.filename).suffix.lower()
        if ext not in allowed:
            return JSONResponse(
                status_code=400,
                content={'error':
                    f'File type {ext} not supported'}
            )

        temp_path = UPLOAD_DIR / f"{uuid.uuid4()}{ext}"
        with open(temp_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)

        if ext == '.pdf':
            try:
                from pdf2image import convert_from_path
                images = convert_from_path(str(temp_path))
                if images:
                    img_path = str(temp_path) + '.png'
                    images[0].save(img_path, 'PNG')
                    temp_path_img = Path(img_path)
                else:
                    return JSONResponse(
                        status_code=400,
                        content={'error': 'Could not read PDF'}
                    )
            except Exception as e:
                return JSONResponse(
                    status_code=400,
                    content={'error': f'PDF error: {str(e)}'}
                )
        else:
            temp_path_img = temp_path

        result = analyze_document(
            str(temp_path_img), documentType)

        return JSONResponse(content={
            'success': True,
            'data': result
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                'success': False,
                'error': str(e)
            }
        )
    finally:
        if temp_path and Path(temp_path).exists():
            try:
                Path(temp_path).unlink()
            except:
                pass
        if temp_path_img and Path(temp_path_img).exists():
            try:
                Path(temp_path_img).unlink()
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('OCR_PORT', 8000))
    uvicorn.run("main:app", host="0.0.0.0",
                port=port, reload=True)
