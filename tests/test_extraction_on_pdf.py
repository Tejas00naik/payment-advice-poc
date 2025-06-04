#!/usr/bin/env python3
"""
Test script to extract invoice data from a real PDF file using both DeepSeek and OpenAI providers
This script will save intermediate data for analysis
"""

import os
import sys
import json
import time
import logging
import shutil
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables
load_dotenv()

# Import the core modules
from src.core.pdf_processor import extract_pdf_data, extract_metadata
from src.core.llm_provider import create_llm, LLMProvider
from src.core.llm_processor import extract_table_structure, normalize_extracted_data
from src.core.schema_processor import create_final_df

def save_results(data, filename, provider_name):
    """Save data to results directory with provider name in filename"""
    results_dir = os.path.join(os.getcwd(), "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    filepath = os.path.join(results_dir, f"{provider_name}_{filename}")
    
    if isinstance(data, str):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(data)
    else:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
    logging.info(f"Saved {filename} to {filepath}")
    return filepath

def test_extraction(pdf_path, provider=None, model_name=None):
    """
    Test the extraction process on a PDF file
    
    Args:
        pdf_path: Path to the PDF file
        provider: LLM provider (deepseek or openai)
        model_name: Model name to use (provider-specific)
    """
    logger = logging.getLogger(__name__)
    
    # Determine provider name for logging and file naming
    if provider:
        provider_str = provider.lower() if isinstance(provider, str) else provider.value
    else:
        # Auto-detect based on available API keys
        if os.getenv("OPENAI_API_KEY"):
            provider_str = "openai"
        elif os.getenv("DEEPSEEK_API_KEY"):
            provider_str = "deepseek"
        else:
            provider_str = "unknown"
    
    if model_name:
        provider_str = f"{provider_str}_{model_name.replace('-', '_')}"
    
    logger.info(f"Testing extraction on {pdf_path} with {provider_str}")
    
    # Step 1: Extract text from PDF
    logger.info("Extracting text from PDF...")
    start_time = time.time()
    text_data = extract_pdf_data(pdf_path)
    pdf_extraction_time = time.time() - start_time
    logger.info(f"PDF text extraction completed in {pdf_extraction_time:.2f} seconds")
    
    # Save raw text for debugging
    text_file = save_results(text_data, "extracted_text.txt", provider_str)
    
    # Step 2: Extract metadata (if needed)
    metadata = extract_metadata(text_data)
    metadata_file = save_results(metadata, "metadata.json", provider_str)
    
    # Step 3: Extract table structure with LLM
    logger.info(f"Extracting table structure with {provider_str}...")
    
    # Start timing
    table_start_time = time.time()
    
    # Extract table structure
    table_data = extract_table_structure(text_data)
    
    # Calculate time
    table_extraction_time = time.time() - table_start_time
    logger.info(f"Table extraction completed in {table_extraction_time:.2f} seconds")
    
    # Save intermediate table results
    table_file = save_results(table_data, "extracted_structure.json", provider_str)
    
    # Step 4: Normalize extracted data
    logger.info(f"Normalizing extracted data with {provider_str}...")
    
    # Start timing
    normalize_start_time = time.time()
    
    # Normalize data
    normalized_items = normalize_extracted_data(table_data, text_data)
    
    # Calculate time
    normalize_time = time.time() - normalize_start_time
    logger.info(f"Normalization completed in {normalize_time:.2f} seconds")
    
    # Save normalized items
    normalized_file = save_results(normalized_items, "normalized_items.json", provider_str)
    
    # Step 5: Create final dataframe
    logger.info("Creating final dataframe...")
    df = create_final_df(normalized_items, metadata)
    
    # Calculate total processing time
    total_time = pdf_extraction_time + table_extraction_time + normalize_time
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    
    # Save results
    if not df.empty:
        # Convert DataFrame to JSON for storage
        df_json = df.to_json(orient="records")
        df_json_parsed = json.loads(df_json)
        df_file = save_results(df_json_parsed, "final_data.json", provider_str)
        
        # Print sample of data
        print(f"\nResults for {provider_str}:")
        print(f"Extracted {len(df)} records")
        print(f"Processing time: {total_time:.2f} seconds")
        print("\nSample data:")
        print(df.head().to_string())
    else:
        logger.warning("No data was extracted")
        print(f"\nResults for {provider_str}: No data was extracted")
    
    # Return extraction times for comparison
    return {
        "provider": provider_str,
        "pdf_extraction_time": pdf_extraction_time,
        "table_extraction_time": table_extraction_time,
        "normalize_time": normalize_time,
        "total_time": total_time,
        "record_count": len(df) if not df.empty else 0,
        "text_file": text_file,
        "table_file": table_file,
        "normalized_file": normalized_file
    }

def main():
    """Run extraction tests with both providers"""
    # Path to the test PDF
    pdf_path = os.path.join(os.getcwd(), "src", "data", "test.pdf")
    
    if not os.path.exists(pdf_path):
        logging.error(f"Test PDF not found at {pdf_path}")
        return
    
    # Create a copy of the PDF in the results directory for reference
    results_dir = os.path.join(os.getcwd(), "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    shutil.copy2(pdf_path, os.path.join(results_dir, "test.pdf"))
    
    results = []
    
    # Test with DeepSeek if API key is available
    if os.getenv("DEEPSEEK_API_KEY"):
        # Force using DeepSeek for the first test
        os.environ["CURRENT_LLM_PROVIDER"] = "deepseek"
        try:
            results.append(test_extraction(pdf_path, provider=LLMProvider.DEEPSEEK, model_name="deepseek-chat"))
        except Exception as e:
            logging.error(f"Error during DeepSeek extraction: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        logging.warning("DeepSeek API key not found, skipping DeepSeek test")
    
    # Test with OpenAI GPT-4.1-mini if API key is available
    if os.getenv("OPENAI_API_KEY"):
        # Force using OpenAI for the second test
        os.environ["CURRENT_LLM_PROVIDER"] = "openai"
        try:
            results.append(test_extraction(pdf_path, provider=LLMProvider.OPENAI, model_name="gpt-4.1-mini"))
        except Exception as e:
            logging.error(f"Error during OpenAI extraction: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        logging.warning("OpenAI API key not found, skipping OpenAI test")
    
    # Compare results
    if len(results) > 1:
        print("\nComparison of extraction times:")
        for result in results:
            print(f"{result['provider']}:")
            print(f"  PDF extraction: {result['pdf_extraction_time']:.2f} seconds")
            print(f"  Table extraction: {result['table_extraction_time']:.2f} seconds")
            print(f"  Normalization: {result['normalize_time']:.2f} seconds")
            print(f"  Total time: {result['total_time']:.2f} seconds")
            print(f"  Records extracted: {result['record_count']}")
            print()
    
    # Save comparison summary
    if results:
        save_results(results, "comparison_summary.json", "all_providers")

if __name__ == "__main__":
    main()
