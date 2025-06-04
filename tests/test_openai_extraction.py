#!/usr/bin/env python3
"""
Test script for OpenAI LLM integration with GPT-4.1-mini model
"""

import os
import sys
import json
import logging
import time
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables
load_dotenv()

# Import our new LLM provider
from src.core.llm_provider import create_llm, LLMProvider

def test_openai_extraction():
    """Test the OpenAI GPT-4.1-mini model for invoice extraction"""
    
    # Create a smaller test prompt to ensure quick response
    test_invoice = """
    Invoice Number: 26401351104458
    Date: 09-MAY-24
    Description: Co-op-8368-Splitit
    Amount Paid: 74.16
    Amount Taken: 74.16
    Amount Remaining: 0.00
    
    Invoice Number: 311209
    Date: 09-MAY-24
    Description: Co-op-8367-Splitit
    Amount Paid: 26.54
    Amount Taken: 26.54
    Amount Remaining: 0.00
    """
    
    # Create extraction prompt
    extraction_prompt = f"""
    You are a payment processing specialist. Extract all invoice information from this payment advice as JSON.
    
    Text from the payment advice:
    ```
    {test_invoice}
    ```
    
    Instructions:
    1. Focus ONLY on identifying and extracting tables containing invoice or payment details
    2. Extract each row of the table as a separate item
    3. Ensure each item has consistent fields (invoice_number, date, description, amount)
    
    Return JSON with an 'invoices' array.
    """
    
    try:
        logging.info("Testing OpenAI GPT-4.1-mini model for invoice extraction")
        
        # Create OpenAI LLM with GPT-4.1-mini model
        llm = create_llm(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4.1-mini", 
            temperature=0,
            json_mode=True,
            max_tokens=1000,
            request_timeout=30.0  # Shorter timeout for testing
        )
        
        if not llm:
            logging.error("Failed to initialize OpenAI LLM")
            return
        
        # Send the prompt
        logging.info(f"Sending extraction prompt to OpenAI (length: {len(extraction_prompt)} chars)")
        start_time = time.time()
        
        # Create message and invoke
        messages = [HumanMessage(content=extraction_prompt)]
        logging.info("Making API call to OpenAI...")
        response = llm.invoke(messages)
        
        elapsed = time.time() - start_time
        logging.info(f"API call completed in {elapsed:.2f} seconds")
        
        # Parse and display response
        if hasattr(response, 'content'):
            logging.info(f"Raw response content: {response.content}")
            
            # Try to parse JSON response
            try:
                parsed_data = json.loads(response.content)
                logging.info(f"Successfully parsed JSON response: {json.dumps(parsed_data, indent=2)}")
                
                # Check for invoices array
                if 'invoices' in parsed_data:
                    invoices = parsed_data['invoices']
                    logging.info(f"Found 'invoices' array with {len(invoices)} items")
                    
                    # Display extracted invoices
                    for i, invoice in enumerate(invoices):
                        print(f"\nInvoice {i+1}:")
                        for key, value in invoice.items():
                            print(f"  {key}: {value}")
                else:
                    logging.warning("Response JSON did not contain 'invoices' array")
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse response as JSON: {str(e)}")
        else:
            logging.error("Unexpected response format, no content attribute found")
    
    except Exception as e:
        logging.error(f"Error during OpenAI extraction test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_openai_extraction()
