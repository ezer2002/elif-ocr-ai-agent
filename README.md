# Elif OCR Service

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-00A393?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Gemini AI](https://img.shields.io/badge/Gemini%20AI-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![License](https://img.shields.io/badge/License-Unlicensed-lightgrey.svg)]()

Elif OCR Service is a Python FastAPI microservice that extracts structured data from uploaded documents for the pet transit platform.
It uses Google Gemini 1.5 Flash first, then falls back to Tesseract OCR and regex parsing when Gemini is unavailable or fails.

## 📚 Table of Contents

- [✨ Features](#-features)
- [🧩 Architecture](#-architecture)
- [🛠️ Tech Stack](#-tech-stack)
- [📋 Prerequisites](#-prerequisites)
- [⚙️ Installation & Setup](#-installation--setup)
- [🌍 Environment Variables](#-environment-variables)
- [🔌 API Reference](#-api-reference)
- [🔁 OCR Pipeline](#-ocr-pipeline)
- [🗂️ Project Structure](#-project-structure)
- [🔗 Integration](#-integration)

## ✨ Features

- 🧠 Intelligent document extraction with Gemini 1.5 Flash for high-quality structured output.
- 🔁 Automatic fallback to Tesseract OCR plus regex parsing if Gemini is unavailable.
- 📄 Supports PDF, JPG, JPEG, and PNG uploads through a REST API.
- 🕒 Detects expired documents and returns a clear `isExpired` flag.
- 🧾 Extracts key fields such as document number, holder name, issue date, expiry date, issuing organization, and document type.
- 🌐 Handles English and French documents.
- 📊 Returns a confidence score, missing fields, warnings, raw text, and the extraction source.
- 🔒 CORS is configured for the Angular frontend and Spring Boot backend used in the larger system.

## 🧩 Architecture

```text
Angular Frontend (http://localhost:4200)
						|
						v
Spring Boot Backend (http://localhost:8087)
						|
						v
		 OCR Service (FastAPI)
						|
	 +--------+---------+
	 |                  |
	 v                  v
Gemini 1.5 Flash   Tesseract OCR
	 |                  |
	 +--------+---------+
						|
						v
	 Structured JSON Response
```

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
| --- | --- | --- |
| API Framework | FastAPI | REST endpoints and request handling |
| ASGI Server | Uvicorn | Local development and production serving |
| AI Extraction | Google Generative AI SDK | Vision-based structured document parsing |
| OCR Engine | pytesseract / Tesseract | Fallback text extraction |
| Image Processing | Pillow | Preprocessing for OCR accuracy |
| PDF Conversion | pdf2image | Converts PDF pages to images |
| Configuration | python-dotenv | Loads environment variables from `.env` |
| Language | Python 3.x | Service implementation |

## 📋 Prerequisites

### Option 1: Docker (Recommended — No Manual Setup)
- Docker Desktop installed
- Access to a Google Gemini API key (and OpenAI key for Level 1 OCR, optional)

### Option 2: Local Python Development
- Python 3.8 or later
- Tesseract OCR binary for Windows
- Poppler for Windows, required by `pdf2image`
- Access to a Google Gemini API key

Useful links:

- Tesseract for Windows: https://github.com/UB-Mannheim/tesseract/wiki
- Poppler for Windows: install a Windows build compatible with `pdf2image`

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd elif-ocr-service
```

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create your `.env` file

```env
GEMINI_API_KEY=your_google_gemini_api_key
OCR_PORT=8000
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### 5. Verify external binaries

Ensure Tesseract and Poppler are installed and available on your machine. If Tesseract is installed in a different location, update `TESSERACT_CMD` accordingly.

### 6. Start the service

Run the app directly:

```bash
python main.py
```

Or start it with Uvicorn:

```bash
uvicorn main:app --reload --port 8000
```

## 🐳 Running with Docker

### Recommended Approach (Zero Local Setup)

Docker eliminates the need for manual Python, Tesseract, and Poppler installation on your machine.

#### First Time Setup

1. **Copy the environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Fill in your API keys in `.env`:**
   ```env
   GEMINI_API_KEY=your_actual_gemini_key_here
   OPENAI_API_KEY=your_actual_openai_key_here (optional)
   OCR_PORT=8000
   ```

3. **Build and start the service:**
   ```bash
   docker compose up --build
   ```

#### Every Time After

```bash
docker compose up
```

#### Stop the service

```bash
docker compose down
```

#### View logs

```bash
docker compose logs -f
```

#### What Happens Inside the Container

- ✅ Python 3.11 environment
- ✅ Tesseract OCR + French language pack
- ✅ Poppler utilities for PDF handling
- ✅ All Python dependencies installed
- ✅ Service runs on `http://localhost:8000`

#### Verify It's Running

Open your browser or use curl:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "ok", "gemini_key": true, "openai_key": false}
```

#### Troubleshooting

- **Port 8000 already in use?**
  - Edit `docker-compose.yml` and change `8000:8000` to `8001:8000` for example
  
- **API keys not working?**
  - Verify they're in `.env` (NOT `.env.example`)
  - Restart the container: `docker compose down && docker compose up`
  
- **View container logs:**
  ```bash
  docker compose logs elif-ocr-agent
  ```

## 🌍 Environment Variables

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `GEMINI_API_KEY` | Yes | None | Google Gemini API key used for primary extraction |
| `OCR_PORT` | No | `8000` | Port used when starting the service from `main.py` |
| `TESSERACT_CMD` | No | `C:\Program Files\Tesseract-OCR\tesseract.exe` | Full path to the Tesseract executable |

## 🔌 API Reference

### `GET /health`

Returns the service status and whether Gemini is configured.

#### Example response

```json
{
	"status": "ok",
	"service": "elif-ocr",
	"gemini_api_configured": true
}
```

### `POST /ocr/analyze`

Accepts a multipart form upload with:

- `file`: PDF, JPG, JPEG, or PNG
- `documentType`: string describing the expected document type

#### Example request

```bash
curl -X POST "http://localhost:8000/ocr/analyze" \
	-F "file=@C:/path/to/document.pdf" \
	-F "documentType=pet passport"
```

#### Example success response

```json
{
	"success": true,
	"data": {
		"documentNumber": "ABC123456",
		"holderName": "Jane Doe",
		"issueDate": "2024-01-15",
		"expiryDate": "2026-01-15",
		"issuingOrganization": "Ministry of Agriculture",
		"detectedDocumentType": "pet passport",
		"confidence": 0.94,
		"rawExtractedText": "...",
		"missingFields": [],
		"isExpired": false,
		"warnings": [],
		"source": "gemini"
	}
}
```

#### Example error response

```json
{
	"success": false,
	"error": "File type .txt not supported"
}
```

## 🔁 OCR Pipeline

1. The API receives an uploaded document and validates the file type.
2. PDFs are converted to images with `pdf2image`; image uploads are processed directly.
3. The service tries Gemini 1.5 Flash first for structured extraction.
4. If Gemini is unavailable, misconfigured, or returns low-quality output, the service falls back to Tesseract OCR.
5. Regex parsing and date normalization refine the fallback output.
6. The final response includes the extracted fields, confidence score, missing fields, expiration status, warnings, and the extraction source.

## 🗂️ Project Structure

```text
elif-ocr-service/
├── main.py
├── ocr_engine.py
├── requirements.txt
├── README.md
├── temp_uploads/
└── .gitattributes
```

## 🔗 Integration

This microservice is designed to work with the main pet transit platform as a dedicated OCR and document intelligence layer. The Angular frontend uploads files, the Spring Boot backend orchestrates the flow, and this service handles the actual extraction work.


