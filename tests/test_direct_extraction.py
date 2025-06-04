#!/usr/bin/env python3
"""
Test script for the fixed direct API extraction
"""

import os
import logging
from dotenv import load_dotenv
from src.core.llm_processor import extract_table_structure

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

def test_extraction():
    """Test the extract_table_structure function with the extracted text"""
    try:
        # Load the extracted text
        with open("extracted_text.txt", "r") as f:
            text_data = f.read()
            logging.info(f"Read {len(text_data)} characters from extracted_text.txt")
        
        # Call the extraction function
        logging.info("Calling extract_table_structure function...")
        result = extract_table_structure(text_data)
        
        # Log and display results
        if result:
            logging.info(f"Successfully extracted {len(result)} items")
            print(f"\nExtracted {len(result)} items:")
            
            # Print the first 5 items
            for i, item in enumerate(result[:5]):
                print(f"{i+1}. {item}")
            
            if len(result) > 5:
                print(f"... and {len(result)-5} more items")
        else:
            logging.error("Extraction returned empty result")
            print("No data extracted")
    
    except Exception as e:
        logging.error(f"Error in test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_extraction()
