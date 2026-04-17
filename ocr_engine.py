import os
import re
import json
import base64
from datetime import datetime
from pathlib import Path
from PIL import Image
import pytesseract
from dotenv import load_dotenv

load_dotenv()

TESSERACT_CMD = os.getenv(
    'TESSERACT_CMD',
    r'C:\Program Files\Tesseract-OCR\tesseract.exe'
)
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def preprocess_image(image_path: str):
    """Enhance image for better OCR — uses only Pillow."""
    try:
        img = Image.open(image_path).convert('L')
        # Increase size for better OCR
        w, h = img.size
        if w < 1000:
            img = img.resize((w * 2, h * 2), Image.LANCZOS)
        return img
    except Exception:
        return Image.open(image_path).convert('L')

def extract_with_tesseract(image_path: str) -> dict:
    """Extract text using Tesseract OCR."""
    try:
        pil_image = preprocess_image(image_path)
        text = pytesseract.image_to_string(
            pil_image,
            lang='eng+fra',
            config='--psm 3 --oem 3'
        )
        return parse_document_text(text)
    except Exception as e:
        return {
            'documentNumber': None,
            'holderName': None,
            'issueDate': None,
            'expiryDate': None,
            'issuingOrganization': None,
            'detectedDocumentType': None,
            'confidence': 0.0,
            'rawExtractedText': '',
            'missingFields': [],
            'isExpired': False,
            'warnings': [f'Tesseract error: {str(e)}'],
            'source': 'tesseract_error'
        }

def extract_with_gemini(image_path: str,
                        document_type: str):
    """Extract structured data using Gemini Vision."""
    try:
        import google.generativeai as genai

        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            return None

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')

        with open(image_path, 'rb') as f:
            image_data = f.read()

        ext = Path(image_path).suffix.lower()
        mime_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
        }
        mime_type = mime_map.get(ext, 'image/jpeg')

        prompt = f"""You are analyzing a pet travel document.
Document type: {document_type}

Extract ALL visible information and return ONLY
a valid JSON object with these exact fields:

{{
  "documentNumber": "extracted number or null",
  "holderName": "full name of holder or null",
  "issueDate": "YYYY-MM-DD format or null",
  "expiryDate": "YYYY-MM-DD format or null",
  "issuingOrganization": "authority name or null",
  "detectedDocumentType": "detected type or null",
  "confidence": number between 0.0 and 1.0,
  "rawExtractedText": "all visible text in document",
  "missingFields": ["fields not found"],
  "isExpired": true or false,
  "warnings": ["any issues found"]
}}

Rules:
- Return ONLY valid JSON, no explanation, no markdown
- Use null for fields not found
- confidence 0.9+ all fields found
- confidence 0.5-0.9 some fields found
- confidence below 0.5 document unclear
- isExpired true if expiryDate is past today"""

        image_part = {
            'mime_type': mime_type,
            'data': base64.b64encode(image_data)
                         .decode('utf-8')
        }

        response = model.generate_content(
            [prompt, image_part])

        text = response.text.strip()

        # Clean markdown if present
        if '```' in text:
            text = re.sub(r'```json?\s*', '', text)
            text = re.sub(r'```\s*', '', text)
            text = text.strip()

        result = json.loads(text)
        result['source'] = 'gemini'
        return result

    except json.JSONDecodeError:
        return None
    except Exception:
        return None

def parse_document_text(text: str) -> dict:
    """Parse raw text to extract document fields."""
    result = {
        'documentNumber': None,
        'holderName': None,
        'issueDate': None,
        'expiryDate': None,
        'issuingOrganization': None,
        'detectedDocumentType': None,
        'confidence': 0.0,
        'rawExtractedText': text,
        'missingFields': [],
        'isExpired': False,
        'warnings': [],
        'source': 'tesseract'
    }

    # Document number
    doc_patterns = [
        r'\b([A-Z]{2,3}[\s\-]?\d{4,10})\b',
        r'\bNo[.:\s]+([A-Z0-9\-]{4,15})\b',
        r'\bNumber[:\s]+([A-Z0-9\-]{4,15})\b',
        r'\b(\d{6,12})\b'
    ]
    for pattern in doc_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result['documentNumber'] = match.group(1)
            break

    # Dates
    date_patterns = [
        r'\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})\b',
        r'\b(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})\b',
    ]
    dates_found = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        dates_found.extend(matches)

    if len(dates_found) >= 2:
        result['issueDate'] = normalize_date(dates_found[0])
        result['expiryDate'] = normalize_date(dates_found[-1])
    elif len(dates_found) == 1:
        result['expiryDate'] = normalize_date(dates_found[0])

    # Check expiry
    if result['expiryDate']:
        try:
            expiry = datetime.strptime(
                result['expiryDate'], '%Y-%m-%d')
            result['isExpired'] = expiry < datetime.now()
            if result['isExpired']:
                result['warnings'].append(
                    'Document is expired')
        except:
            pass

    # Name detection
    name_patterns = [
        r'(?:Name|Holder|Owner)[:\s]+([A-Z][a-z]+'
        r'\s+[A-Z][a-z]+)',
    ]
    for pattern in name_patterns:
        match = re.search(pattern, text)
        if match:
            result['holderName'] = match.group(1).title()
            break

    # Calculate confidence
    fields_to_check = [
        'documentNumber', 'holderName',
        'issueDate', 'expiryDate', 'issuingOrganization'
    ]
    result['missingFields'] = [
        f for f in fields_to_check if not result.get(f)]
    found = len(fields_to_check) - len(result['missingFields'])
    result['confidence'] = round(found / len(fields_to_check), 2)

    return result

def normalize_date(date_str: str) -> str:
    """Normalize date to YYYY-MM-DD."""
    formats = [
        '%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y',
        '%Y/%m/%d', '%Y-%m-%d', '%m/%d/%Y',
        '%d/%m/%y',
    ]
    for fmt in formats:
        try:
            return datetime.strptime(
                date_str.strip(), fmt
            ).strftime('%Y-%m-%d')
        except:
            continue
    return date_str

def analyze_document(image_path: str,
                     document_type: str = 'UNKNOWN') -> dict:
    """Try Gemini first, fallback to Tesseract."""
    gemini_result = extract_with_gemini(
        image_path, document_type)

    if (gemini_result
            and isinstance(gemini_result, dict)
            and gemini_result.get('confidence', 0) > 0.4):
        return gemini_result

    # Fallback
    result = extract_with_tesseract(image_path)
    result['source'] = 'tesseract_fallback'
    return result
