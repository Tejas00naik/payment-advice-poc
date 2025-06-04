#!/usr/bin/env python3
"""
Main entry point for the Automated Payment Advice Extraction System.
"""
import os
import sys
import argparse
import logging
import json
import pandas as pd
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
import signal

# Add the project root to sys.path to make imports work correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the core modules
from core.pdf_processor import extract_pdf_data, extract_metadata
from core.llm_processor import extract_table_structure, normalize_extracted_data
from core.schema_processor import FINAL_COLUMNS, create_final_df

# Load environment variables
load_dotenv()

# Setup logging
def setup_logging():
    """Setup logging configuration"""
    log_dir = os.path.join(os.getcwd(), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, "extraction.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Automated Payment Advice Extraction System")
    
    # Input sources
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--file", help="Path to a single file to process")
    group.add_argument("-d", "--directory", help="Path to directory containing files to process")
    
    # Processing options
    parser.add_argument("-t", "--type", choices=["pdf", "excel", "text"], default="pdf",
                        help="Type of input for single file processing")
    parser.add_argument("-w", "--workers", type=int, default=4,
                        help="Number of worker threads for batch processing")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable caching of results")
    
    # Output options
    parser.add_argument("-o", "--output", default="extracted_data.csv",
                        help="Output file path")
    
    return parser.parse_args()


def process_single_file(file_path: str, timeout: int = 300) -> pd.DataFrame:
    """Process a single PDF file using LangChain extraction
    
    Args:
        file_path: Path to the PDF file
        timeout: Maximum time in seconds for LLM calls
        
    Returns:
        DataFrame with extracted data
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing PDF file: {file_path}")
    
    # Extract text from PDF
    logger.info("Extracting text from PDF...")
    text_data = extract_pdf_data(file_path)
    
    # Save raw text for debugging
    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(text_data)
    logger.info("Raw text data saved to extracted_text.txt")
    
    # Extract metadata
    metadata = extract_metadata(text_data)
    logger.info(f"Extracted metadata: {metadata}")
    
    # Use signal to handle timeouts
    def timeout_handler(signum, frame):
        raise TimeoutError("LLM processing timed out")
    
    # Step 1: Extract table structure with timeout
    logger.info(f"Step 1: Extracting table structure with LLM (with {timeout}s timeout)...")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        # First step: Extract only the tables from the PDF text
        table_data = extract_table_structure(text_data)
        signal.alarm(0)  # Cancel the alarm
        
        # Save intermediate table results
        with open("extracted_structure.json", "w") as f:
            json.dump(table_data, f, indent=2)
        logger.info("Extracted table structure saved to extracted_structure.json")
        
    except TimeoutError:
        signal.alarm(0)  # Cancel the alarm
        logger.error(f"Table extraction timed out after {timeout} seconds")
        return pd.DataFrame(columns=FINAL_COLUMNS)
    except Exception as e:
        signal.alarm(0)  # Cancel the alarm
        logger.error(f"Table extraction failed: {e}")
        return pd.DataFrame(columns=FINAL_COLUMNS)
    
    # Step 2: Normalize data using both table structure and full text
    logger.info("Step 2: Normalizing data with both table structure and full text (60s timeout)...")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)  # 60 second timeout for normalization
    try:
        # Second step: Use both table data AND full text for normalization
        normalized_items = normalize_extracted_data(table_data, text_data)
        signal.alarm(0)  # Cancel the alarm
        
        # Save normalized items
        with open("normalized_items.json", "w") as f:
            json.dump(normalized_items, f, indent=2)
        logger.info("Normalized items saved to normalized_items.json")
        
        # Ensure normalized_items is a list of dictionaries
        if isinstance(normalized_items, dict):
            # If we got a single invoice as a dictionary, wrap it in a list
            normalized_items = [normalized_items]
            logger.info("Converted single dictionary to list for DataFrame creation")
        
        # Create DataFrame with final schema
        df = create_final_df(normalized_items, metadata)
        logger.info(f"Successfully extracted {len(df)} records")
        return df
        
    except TimeoutError:
        signal.alarm(0)  # Cancel the alarm
        logger.error("Normalization timed out")
        return pd.DataFrame(columns=FINAL_COLUMNS)
    except Exception as e:
        signal.alarm(0)  # Cancel the alarm
        logger.error(f"Normalization failed: {e}")
        return pd.DataFrame(columns=FINAL_COLUMNS)

def process_directory_batch(directory: str, workers: int = 4) -> pd.DataFrame:
    """Process all PDF files in a directory using multiprocessing
    
    Args:
        directory: Directory path containing PDF files
        workers: Number of worker processes
        
    Returns:
        Combined DataFrame with results from all files
    """
    import glob
    from concurrent.futures import ProcessPoolExecutor
    
    logger = logging.getLogger(__name__)
    
    # Find all PDF files
    pdf_files = glob.glob(os.path.join(directory, "**", "*.pdf"), recursive=True)
    logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {directory}")
        return pd.DataFrame(columns=FINAL_COLUMNS)
    
    # Process files in parallel
    all_dfs = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for i, df in enumerate(executor.map(process_single_file, pdf_files)):
            logger.info(f"Completed file {i+1}/{len(pdf_files)}")
            if not df.empty:
                all_dfs.append(df)
    
    # Combine all results
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame(columns=FINAL_COLUMNS)

def main():
    """Main function to orchestrate the extraction process."""
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Automated Payment Advice Extraction System")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Process input source
    if args.file:
        logger.info(f"Processing single file: {args.file}")
        df = process_single_file(args.file)
    elif args.directory:
        logger.info(f"Processing directory: {args.directory}")
        df = process_directory_batch(args.directory, args.workers)
    
    # Save results
    if not df.empty:
        logger.info(f"Extracted {len(df)} records")
        df.to_csv(args.output, index=False)
        logger.info(f"Data saved to {args.output}")
        
        # Display summary
        print("\nExtraction Summary:")
        print(f"Total records: {len(df)}")
        print(f"Unique invoices: {df['Invoice number'].nunique()}")
        print(f"Total amount: {df['Amount paid'].sum():,.2f} {df['Currency'].iloc[0] if not df['Currency'].isnull().all() else ''}")
        
        # Show sample of data
        print("\nSample of extracted data:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 120)
        print(df.head().to_string())
        
        return df
    else:
        logger.warning("No data was extracted")
        print("No data was extracted from the input source.")
        return pd.DataFrame(columns=FINAL_COLUMNS)


def process_email(email_dir: Optional[str] = None):
    """Process emails (placeholder for future implementation)"""
    logger = logging.getLogger(__name__)
    logger.info("Email processing is not yet implemented")
    print("Email processing is planned for future implementation.")
    return pd.DataFrame(columns=FINAL_COLUMNS)


if __name__ == "__main__":
    main()
