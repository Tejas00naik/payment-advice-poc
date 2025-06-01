#!/usr/bin/env python3
"""
Test script for LangChain-based payment advice extraction with DeepSeek API.
"""
import os
import sys
import logging
import json
import pandas as pd
import signal
from dotenv import load_dotenv

# Import the extraction functions from our codebase
from pdf_processor import extract_pdf_data, extract_metadata
from llm_processor import extract_table_structure, normalize_extracted_data
from schema_processor import create_final_df

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def process_pdf(pdf_path, timeout=300):
    """Process a PDF file using LangChain with DeepSeek API"""
    if not os.path.exists(pdf_path):
        logging.error(f"PDF file not found: {pdf_path}")
        return None
    
    logging.info(f"Processing PDF file: {pdf_path}")
    
    # Extract text from PDF
    logging.info("Extracting text from PDF...")
    text_data = extract_pdf_data(pdf_path)
    
    # Save raw text for debugging
    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(text_data)
    logging.info("Raw text data saved to extracted_text.txt")
    
    # Extract metadata
    metadata = extract_metadata(text_data)
    logging.info(f"Extracted metadata: {metadata}")
    
    # Use signal to handle timeouts
    def timeout_handler(signum, frame):
        raise TimeoutError("LLM processing timed out")
    
    # Extract table structure with timeout
    logging.info(f"Extracting table structure with LLM (with {timeout}s timeout)...")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        extracted_structure = extract_table_structure(text_data)
        signal.alarm(0)  # Cancel the alarm
        
        # Save intermediate results
        with open("extracted_structure.json", "w") as f:
            json.dump(extracted_structure, f, indent=2)
        logging.info("Extracted structure saved to extracted_structure.json")
        
    except TimeoutError:
        signal.alarm(0)  # Cancel the alarm
        logging.error(f"Table extraction timed out after {timeout} seconds")
        return None
    except Exception as e:
        signal.alarm(0)  # Cancel the alarm
        logging.error(f"Table extraction failed: {e}")
        return None
    
    # Normalize extracted data with timeout
    logging.info("Normalizing extracted data (with 60s timeout)...")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)  # 60 second timeout for normalization
    try:
        normalized_items = normalize_extracted_data(extracted_structure)
        signal.alarm(0)  # Cancel the alarm
        
        # Save normalized items
        with open("normalized_items.json", "w") as f:
            json.dump(normalized_items, f, indent=2)
        logging.info("Normalized items saved to normalized_items.json")
        
        # Create DataFrame with final schema
        df = create_final_df(normalized_items, metadata)
        return df
        
    except TimeoutError:
        signal.alarm(0)  # Cancel the alarm
        logging.error("Normalization timed out")
        return None
    except Exception as e:
        signal.alarm(0)  # Cancel the alarm
        logging.error(f"Normalization failed: {e}")
        return None

def main():
    # Define the PDF file to process
    pdf_file = "test.pdf"
    
    # Check if file exists
    if not os.path.exists(pdf_file):
        logging.error(f"Test PDF file not found: {pdf_file}")
        return
    
    # Process the PDF
    df = process_pdf(pdf_file)
    
    if df is not None and not df.empty:
        # Display summary
        logging.info(f"Successfully extracted {len(df)} records")
        
        # Save to CSV
        output_file = "deepseek_extracted_data.csv"
        df.to_csv(output_file, index=False)
        logging.info(f"Results saved to {output_file}")
        
        # Display sample
        logging.info("\nSample of extracted data:")
        print(df.head())
    else:
        logging.error("Extraction failed or no data was extracted")

if __name__ == "__main__":
    main()
