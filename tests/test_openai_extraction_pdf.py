#!/usr/bin/env python3
"""
Test script for invoice extraction from PDF using OpenAI GPT-4.1-mini.
This script will process the sample PDF and save all intermediate data.
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add project root to path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Import the actual functions from the codebase - using the exact implementations
from src.core.pdf_processor import extract_pdf_data, extract_metadata
from src.core.llm_processor import extract_table_structure, normalize_extracted_data
from src.core.schema_processor import create_final_df
from src.core.llm_provider import create_llm, LLMProvider

def main():
    """Process the test PDF file using OpenAI GPT-4.1-mini model"""
    logger = logging.getLogger(__name__)
    
    # Create results directory
    results_dir = Path("results/openai")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Path to test PDF
    pdf_path = Path("src/data/test.pdf")
    if not pdf_path.exists():
        logger.error(f"Test PDF not found at {pdf_path}")
        return
    
    # Force using OpenAI
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    
    # Step 1: Extract text from PDF
    logger.info(f"Processing PDF file: {pdf_path}")
    start_time = time.time()
    text_data = extract_pdf_data(str(pdf_path))
    pdf_time = time.time() - start_time
    logger.info(f"PDF text extraction completed in {pdf_time:.2f} seconds")
    
    # Save extracted text
    with open(results_dir / "extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(text_data)
    logger.info(f"Raw text saved to {results_dir / 'extracted_text.txt'}")
    
    # Step 2: Extract metadata
    logger.info("Extracting metadata...")
    metadata = extract_metadata(text_data)
    with open(results_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {results_dir / 'metadata.json'}")
    
    # Step 3: Extract table structure using OpenAI
    logger.info("Extracting table structure with OpenAI GPT-4.1-mini...")
    
    # Start timing
    table_start_time = time.time()
    
    # Extract table structure
    table_data = extract_table_structure(text_data)
    
    # Calculate time
    table_time = time.time() - table_start_time
    logger.info(f"Table extraction completed in {table_time:.2f} seconds")
    
    # Save table data
    with open(results_dir / "extracted_structure.json", "w", encoding="utf-8") as f:
        json.dump(table_data, f, indent=2)
    logger.info(f"Table structure saved to {results_dir / 'extracted_structure.json'}")
    
    # Step 4: Normalize extracted data
    logger.info("Normalizing extracted data...")
    normalize_start_time = time.time()
    normalized_items = normalize_extracted_data(table_data)
    normalize_time = time.time() - normalize_start_time
    logger.info(f"Normalization completed in {normalize_time:.2f} seconds")
    
    # Save normalized data
    with open(results_dir / "normalized_items.json", "w", encoding="utf-8") as f:
        json.dump(normalized_items, f, indent=2)
    logger.info(f"Normalized data saved to {results_dir / 'normalized_items.json'}")
    
    # Step 5: Create final dataframe
    logger.info("Creating final dataframe...")
    
    # Ensure normalized_items is a list of dictionaries
    if isinstance(normalized_items, dict):
        # If we got a single invoice as a dictionary, wrap it in a list
        normalized_items = [normalized_items]
    
    df = create_final_df(normalized_items, metadata)
    
    # Save and display results
    if not df.empty:
        # Save to CSV
        csv_path = results_dir / "final_data.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Final data saved to {csv_path}")
        
        # Save to JSON for inspection
        json_data = json.loads(df.to_json(orient="records"))
        with open(results_dir / "final_data.json", "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
            
        # Print summary
        total_time = pdf_time + table_time + normalize_time
        print("\n========== OpenAI GPT-4.1-mini Extraction Results ==========")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"  - PDF extraction: {pdf_time:.2f} seconds")
        print(f"  - Table extraction (LLM): {table_time:.2f} seconds")
        print(f"  - Data normalization: {normalize_time:.2f} seconds")
        print(f"Records extracted: {len(df)}")
        print(f"\nSample data (first {min(5, len(df))} records):")
        print(df.head().to_string())
        print("\nAll intermediate data and results saved to:", results_dir)
    else:
        logger.warning("No data was extracted")
        print("No data was extracted from the PDF.")

if __name__ == "__main__":
    main()
