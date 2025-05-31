# Automated Payment Advice Extraction System

A system to extract invoice data (ID, amount, currency, date) from emails and their attachments into a unified pandas DataFrame, handling variable formats.

## Features

- Email ingestion from various sources
- PDF and image attachment processing
- Text extraction using OCR when needed
- Structured data extraction using pattern matching
- Unified DataFrame output for further analysis

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your credentials
```

3. Run the extraction:
```bash
python main.py
```

## Project Structure

- `main.py`: Entry point for the application
- `email_processor.py`: Handles email fetching and parsing
- `document_processor.py`: Processes PDF and image attachments
- `data_extractor.py`: Extracts structured data from text
- `utils.py`: Utility functions
- `config.py`: Configuration settings

## Requirements

- Python 3.8+
- Tesseract OCR (for image processing)
- Ghostscript (for PDF processing)
