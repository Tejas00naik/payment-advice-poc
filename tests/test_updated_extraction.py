#!/usr/bin/env python3
"""
Test the updated extraction function with 60s timeout and improved text handling
"""

import os
import sys
import logging
import time
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

def test_extract_table():
    """Test the updated extract_table_structure function with real data"""
    # Import here to ensure we're using the updated version
    from src.core.llm_processor import extract_table_structure
    
    try:
        # Read the extracted text data
        with open("extracted_text.txt", "r") as f:
            text_data = f.read()
            logging.info(f"Read {len(text_data)} characters from extracted_text.txt")
        
        # Call the extraction function
        logging.info("Calling extract_table_structure...")
        start_time = time.time()
        
        table_data = extract_table_structure(text_data)
        
        elapsed = time.time() - start_time
        logging.info(f"Total extraction completed in {elapsed:.2f} seconds")
        
        # Check the results
        if not table_data:
            logging.error("No invoice data was extracted")
            return
            
        # Handle both possible return formats (list or dict with invoices key)
        if isinstance(table_data, dict) and 'invoices' in table_data:
            invoices = table_data['invoices']
            logging.info(f"Successfully extracted {len(invoices)} invoice items from 'invoices' array")
        else:
            invoices = table_data
            logging.info(f"Successfully extracted {len(invoices)} invoice items")
        
        # Display the first few extracted invoices
        for i, invoice in enumerate(invoices[:3]):
            print(f"\nInvoice {i+1}:")
            for key, value in invoice.items():
                print(f"  {key}: {value}")
        
        if len(invoices) > 3:
            print(f"\n... and {len(invoices)-3} more invoices")
    
    except Exception as e:
        logging.error(f"Error during extraction test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_extract_table()
