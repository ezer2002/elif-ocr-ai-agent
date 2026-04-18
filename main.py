import os
import uuid
import shutil
from datetime import datetime
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

REQUIRED_FIELDS = [
    'documentNumber',
    'holderName',
    'issueDate',
    'expiryDate',
    'issuingOrganization',
]


@app.on_event("startup")
async def startup_test():
    import logging

    logger = logging.getLogger("startup")

    # Test OpenAI
    try:
        import openai
        openai_key = os.getenv('OPENAI_API_KEY')
        logger.info(f"[STARTUP] OpenAI key: {bool(openai_key)}")
        if openai_key:
            try:
                client = openai.OpenAI(api_key=openai_key)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    max_tokens=10,
                    messages=[
                        {
                            "role": "user",
                            "content": "Reply with exactly: READY"
                        }
                    ],
                )
                logger.info(f"[STARTUP] OpenAI test: {response.choices[0].message.content.strip()}")
            except Exception as e:
                logger.error(f"[STARTUP] OpenAI ERROR: {e}")
    except ImportError:
        logger.warning("[STARTUP] OpenAI library not installed")

    # Test Gemini
    try:
        from google import genai as genai_client
        gemini_key = os.getenv('GEMINI_API_KEY')
        logger.info(f"[STARTUP] Gemini key: {bool(gemini_key)}")
        if gemini_key:
            try:
                client = genai_client.Client(api_key=gemini_key)
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=["Reply with exactly: READY"]
                )
                logger.info(f"[STARTUP] Gemini test: {response.text.strip()}")
            except Exception as e:
                logger.error(f"[STARTUP] Gemini ERROR: {e}")
    except ImportError:
        logger.warning("[STARTUP] Gemini library not installed")

    if not (os.getenv('OPENAI_API_KEY') or os.getenv('GEMINI_API_KEY')):
        logger.warning("[STARTUP] No API keys configured - Tesseract only mode")



def merge_page_results(results: list[dict]) -> dict:
    valid_results = [r for r in results if isinstance(r, dict)]
    if not valid_results:
        return {
            'isRelevantDocument': True,
            'documentNumber': None,
            'holderName': None,
            'issueDate': None,
            'expiryDate': None,
            'issuingOrganization': None,
            'detectedDocumentType': None,
            'confidence': 0.0,
            'rawExtractedText': '',
            'missingFields': REQUIRED_FIELDS.copy(),
            'isExpired': False,
            'documentQuality': 'POOR',
            'warnings': ['Could not extract data from any PDF page.'],
            'rejectionReason': None,
            'source': 'unavailable',
        }

    relevant = [r for r in valid_results if r.get('isRelevantDocument') is not False]
    if not relevant:
        return valid_results[0]

    def score(item: dict) -> tuple:
        found_count = sum(1 for field in REQUIRED_FIELDS if item.get(field))
        return (found_count, float(item.get('confidence') or 0.0))

    ranked = sorted(relevant, key=score, reverse=True)
    merged = dict(ranked[0])

    for field in [
        'documentNumber', 'holderName', 'petName',
        'issueDate', 'expiryDate', 'issuingOrganization',
        'detectedDocumentType', 'rawExtractedText',
        'documentQuality', 'rejectionReason'
    ]:
        if merged.get(field):
            continue
        for item in ranked[1:]:
            if item.get(field):
                merged[field] = item.get(field)
                break

    warnings = []
    for item in ranked:
        for warning in (item.get('warnings') or []):
            if warning not in warnings:
                warnings.append(warning)
    merged['warnings'] = warnings

    merged['source'] = 'gemini' if any(
        item.get('source') == 'gemini' for item in ranked
    ) else (merged.get('source') or 'tesseract_fallback')

    merged['missingFields'] = [
        field for field in REQUIRED_FIELDS if not merged.get(field)
    ]

    merged['isRelevantDocument'] = True
    merged['confidence'] = round(max(
        float(item.get('confidence') or 0.0) for item in ranked
    ), 2)

    expiry_value = merged.get('expiryDate')
    if expiry_value:
        try:
            expiry_dt = datetime.strptime(expiry_value, '%Y-%m-%d')
            merged['isExpired'] = expiry_dt < datetime.now()
        except Exception:
            merged['isExpired'] = bool(
                any(item.get('isExpired') for item in ranked)
            )
    else:
        merged['isExpired'] = False

    if not merged.get('documentQuality'):
        conf = float(merged.get('confidence') or 0.0)
        merged['documentQuality'] = (
            'GOOD' if conf >= 0.85
            else 'MEDIUM' if conf >= 0.40
            else 'POOR'
        )

    return merged

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
                    page_paths = []
                    for idx, image in enumerate(images, start=1):
                        img_path = UPLOAD_DIR / f"{temp_path.stem}_page_{idx}.png"
                        image.save(str(img_path), 'PNG')
                        page_paths.append(img_path)
                    temp_path_img = page_paths
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

        if isinstance(temp_path_img, list):
            page_results = []
            for page_path in temp_path_img:
                page_results.append(
                    analyze_document(str(page_path), documentType)
                )
            result = merge_page_results(page_results)
        else:
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
        if isinstance(temp_path_img, list):
            for path_item in temp_path_img:
                try:
                    if Path(path_item).exists():
                        Path(path_item).unlink()
                except:
                    pass
        elif temp_path_img and Path(temp_path_img).exists():
            try:
                Path(temp_path_img).unlink()
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('OCR_PORT', 8000))
    uvicorn.run("main:app", host="0.0.0.0",
                port=port, reload=True)
