# Payment Advice Extraction System

A robust system to extract invoice data from payment advice documents (PDFs, Excel) into a structured pandas DataFrame, with intelligent handling of variable formats using a hybrid extraction approach.

## System Architecture

```
Input Source → PDF/Excel/Text → Extract Raw Data → Extract Metadata → 
LLM Normalization → Schema Enforcement → Validation → Final DataFrame
```

## Features

- **PDF Expertise**: Camelot handles digital PDF tables, pdfplumber for fallback text extraction
- **LLM Efficiency**: Processes only extracted data, with schema enforcement
- **Metadata Propagation**: Header data applied to all line items
- **Production-Ready**: Parallel processing, caching, validation, error handling

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Install Poppler for Camelot:
   ```bash
   # macOS
   brew install poppler
   
   # Ubuntu/Debian
   apt-get install poppler-utils
   ```

3. Set up environment variables:
   ```bash
   # Create .env file with your OpenAI API key
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   ```

## Usage

### Command Line Interface

Process a single file:
```bash
python main.py -f /path/to/payment_advice.pdf
```

Process a directory of files:
```bash
python main.py -d /path/to/directory
```

Options:
- `-t, --type`: Input type (pdf, excel, text) - default: pdf
- `-w, --workers`: Number of parallel workers - default: 4
- `--no-cache`: Disable caching
- `-o, --output`: Output file path - default: extracted_data.csv

### Python API

```python
from processor import process_payment_advice

# Process a single file
df = process_payment_advice('path/to/file.pdf', 'pdf')

# Save results
df.to_csv('results.csv', index=False)
```

## Project Modules

- `main.py`: Command-line interface and entry point
- `processor.py`: Main processing orchestration
- `pdf_processor.py`: PDF extraction with Camelot and pdfplumber
- `llm_processor.py`: LLM-based normalization with OpenAI
- `schema_processor.py`: Schema enforcement and validation
- `optimization.py`: Caching and parallel processing
- `test_extraction.py`: Test script for the system

## Output Schema

The extraction produces a DataFrame with the following columns:

- Payment date
- Payment advice number
- Invoice number
- Invoice date
- Customer name
- Customer id (SAP id)
- UTR
- Doc type
- Doc number
- Amount paid
- TDS deducted
- Currency
- PA link

## Testing

Run the test script to verify your setup:
```bash
python test_extraction.py
```

## Requirements

- Python 3.8+
- OpenAI API key
- Poppler (for Camelot)
- Pandas, pdfplumber, and other dependencies in requirements.txt
