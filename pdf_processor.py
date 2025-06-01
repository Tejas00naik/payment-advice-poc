#!/usr/bin/env python3
"""
PDF Processing Module for Payment Advice Extraction System.
Handles extraction of text data from PDF files using pdfplumber.
"""
import pdfplumber
import pandas as pd
import os
import logging


def extract_pdf_data(pdf_path: str) -> str:
    """
    Extract text from PDF using pdfplumber
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text content as a string
    """
    # Check if file exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"--- PAGE {i+1} ---\n{page_text}\n\n"
            
            if not text.strip():
                logging.warning(f"No text extracted from {pdf_path}")
                return ""
                
            return text
    except Exception as e:
        logging.error(f"PDF text extraction failed: {e}")
        raise ValueError(f"Could not extract text from {pdf_path}")


def extract_metadata(text: str) -> dict:
    """
    Extract common header fields using regex
    """
    import re
    
    metadata = {
        "payment_advice_no": None,
        "payment_date": None,
        "customer_name": None,
        "currency": None
    }
    
    # Payment advice number patterns
    payment_no_patterns = [
        r"Payment[#\s]*(\d+)",
        r"Payment[- ]?Advice[- ]?(?:No|Number)[.:\s]*(\d+)",
        r"Advice[- ]?(?:No|Number)[.:\s]*(\d+)",
        r"PA[- ]?(?:No|Number)[.:\s]*(\d+)",
    ]
    
    # Payment date patterns
    date_patterns = [
        r"Payment date:\s*(\d{2}-[A-Z]{3}-\d{4})",
        r"Payment date:\s*(\d{2}/\d{2}/\d{4})",
        r"Date:\s*(\d{2}[/-][A-Z]{3}[/-]\d{4})",
        r"Date:\s*(\d{2}[/-]\d{2}[/-]\d{4})",
    ]
    
    # Customer name patterns
    customer_patterns = [
        r"Remittance Advice - (.+?) Payment#",
        r"(?:Customer|Client)[- ]?(?:Name|ID):\s*(.+?)(?:\n|$)",
        r"(?:Paid to|Payee):\s*(.+?)(?:\n|$)",
    ]
    
    # Currency patterns
    currency_patterns = [
        r"Payment currency:\s*([A-Z]{3})",
        r"Currency:\s*([A-Z]{3})",
        r"Amount \(([A-Z]{3})\)",
    ]
    
    # Try all patterns for each field
    for field, patterns in [
        ("payment_advice_no", payment_no_patterns),
        ("payment_date", date_patterns),
        ("customer_name", customer_patterns),
        ("currency", currency_patterns)
    ]:
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata[field] = match.group(1).strip()
                break
    
    return metadata


def get_first_page_text(pdf_path: str) -> str:
    """
    Extract text from first page for metadata extraction
    """
    with pdfplumber.open(pdf_path) as pdf:
        return pdf.pages[0].extract_text()
