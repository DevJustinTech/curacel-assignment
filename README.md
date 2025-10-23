# Curacel Claims QA Service

A FastAPI microservice for processing medical insurance claim documents using OCR, with structured data extraction and Q&A capabilities.

## Overview

This service processes scanned/photographed insurance claim documents to verify treatments, medications, and other provided services. It extracts structured data and provides a Q&A interface for querying the extracted information.

### Key Features
- Document processing (PDF and images)
- Structured data extraction
- Medical claim information parsing
- Q&A capability
- Health monitoring

## Technical Approach

### 1. Document Processing Pipeline
- Uses Tesseract OCR for text extraction
- PDF support via pdf2image for multi-page documents
- Robust error handling and validation
- Supports both image (PNG, JPEG) and PDF formats

### 2. Data Extraction
Implements specialized extractors for:
- Patient & Member Names (context-aware, two-word format)
- Ages (with DOB fallback)
- Medications (name, dosage, quantity)
- Procedures (filtered for medical relevance)
- Diagnoses (keyword-based)
- Admission Details (with date parsing)
- Total Amounts (with currency normalization)

### 3. API Endpoints
- `POST /extract`: Process documents and extract structured data
- `POST /ask`: Query extracted information (with 2s delay)
- `GET /health`: Service health monitoring



## Implementation Decisions & Assumptions

### Name Extraction
- Patient and member names are treated as separate entities
- Names are normalized to exactly two words (first and last name)
- Context-aware extraction to avoid mixing with facility names
- Explicit label matching (e.g., "patient name:", "member name:")

### Medication Processing
- Strips product codes
- Parses dosage patterns (e.g., "100mg", "500 ml")
- Identifies quantities and units
- Normalizes formatting

### Amount Detection
- Supports both ₦ and NGN formats
- Normalizes to ₦ format
- Prioritizes labeled amounts
- Falls back to largest amount heuristic

### Procedures
- Filters out form labels (ending with colon)
- Removes numeric-only entries
- Requires medical context keywords
- Deduplicates while preserving order

### Assumptions and Decisions Made During Implementation

#### General
- Input documents are medical insurance claim sheets or similar forms containing patient, diagnosis, and billing details.
- OCR text accuracy depends on image clarity, lighting, and font quality.
- Only English-language documents are supported.
- Each uploaded file is assumed to represent one claim document (not multiple claims).

#### Design & Architecture
- FastAPI was chosen for its performance, readability, and async capabilities.
- The system is designed as a lightweight, local microservice, with no external API calls.
- Tesseract OCR handles text recognition offline.
- pdf2image converts PDFs into images to support OCR on all pages.
- The `/ask` endpoint simulates a Q&A engine by searching extracted text contextually and includes a 2-second delay for realism.

#### Data Extraction Logic
- Names are extracted using clear context cues like “Patient Name” or “Member Name”.
- Ages and birthdates use numeric and date pattern recognition.
- Diagnoses, medications, and procedures are extracted using rule-based matching and keyword filtering.
- Admission and discharge dates are parsed from recognized date formats.
- Total amounts are detected using ₦ or NGN prefixes and normalized to ₦ format.
- Extraction logic prioritizes labeled fields; in ambiguous cases, nearest contextual matches are chosen.

#### Error Handling & Validation
- OCR or PDF processing errors return descriptive 400-level errors.
- Invalid or empty uploads are rejected gracefully.
- Structured validation ensures extracted values fit expected types and formats.

#### Performance & Scalability
- The service processes documents synchronously, ideal for testing or low-volume environments.
- Data is stored in-memory, cleared after each request.
- Future scalability options include async task queues and persistent storage.

## Setup Instructions

### Prerequisites
1. Python 3.8+
2. Tesseract OCR installed on your system
   ```bash
   # Windows (download installer):
   https://github.com/UB-Mannheim/tesseract/wiki

   # macOS (Homebrew):
   brew install tesseract

   # Linux (Debian/Ubuntu):
   sudo apt-get install tesseract-ocr
   ```
   - On Windows, ensure `C:\Program Files\Tesseract-OCR\` is on your PATH.
   - On macOS/Linux, confirm `tesseract` is available via `which tesseract`.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/DevJustinTech/curacel-assignment
   cd curacel-assignment
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Verify the Tesseract executable path:
   - Windows default: `C:\Program Files\Tesseract-OCR\tesseract.exe` (update `assignment.py` if different).
   - macOS/Linux: check `which tesseract` for the installed binary location.

### Running the Service

1. Start the server:
   ```bash
   uvicorn assignment:app --reload --port 8000
   ```

2. Access the API:
   - Swagger UI: http://localhost:8000/docs
   - API endpoints:
     - POST http://localhost:8000/extract
     - POST http://localhost:8000/ask
     - GET http://localhost:8000/health



### Sample Request
```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.pdf"
```

## Dependencies

```plaintext
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
pillow==10.1.0
pytesseract==0.3.10
pdf2image==1.16.3
python-dotenv==1.0.0
pydantic==2.4.2
```

## Error Handling

The service includes comprehensive error handling for:
- Invalid file formats
- OCR processing failures
- Data extraction errors
- Document not found scenarios
- Invalid request formats

## Limitations & Future Improvements

1. OCR Accuracy
   - Could be improved with image preprocessing
   - Multiple OCR engine support

2. Data Extraction
   - More comprehensive medical terminology
   - Better handling of varied document formats
   - Machine learning-based extraction

3. Performance
   - Caching for frequent queries
   - Batch processing support
   - Async processing for large documents

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request
