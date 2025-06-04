#!/usr/bin/env python3
"""
Minimal test script for DeepSeek API with very small input
"""

import os
import time
import json
import logging
import requests
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

def minimal_api_test():
    """Send a very minimal request to the DeepSeek API"""
    # Get API key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        logging.error("DEEPSEEK_API_KEY not found in environment variables")
        return
    
    # Use a larger prompt similar to what we might use in real extraction
    larger_prompt = """
    You are a payment processing specialist. Extract all table information from this payment advice as JSON.
    
    Text from the payment advice:
    ```
    Invoice Number: Invoice description Amount Paid
    Date: Taken Remaining
    09-MAY-
    26401351104458 Co-op-8368-Splitit 74.16 74.16 0.00
    24
    09-MAY-
    311209 Co-op-8367-Splitit 26.54 26.54 0.00
    24
    09-MAY-
    26401351104459 Co-op-8366-Splitit 89.11 89.11 0.00
    24
    09-MAY-
    311210 Co-op-8365-Splitit 425.25 425.25 0.00
    24
    ```
    
    Instructions:
    1. Focus ONLY on identifying and extracting tables containing invoice or payment details
    2. Extract each row of the table as a separate item
    3. Do NOT extract metadata like payment date, customer, etc. at this stage
    4. Ensure each item has consistent fields (invoice_number, date, description, amount)
    
    Return JSON with an 'invoices' array.
    """
    
    logging.info(f"Sending larger prompt to DeepSeek API (length: {len(larger_prompt)} chars)")
    start_time = time.time()
    
    try:
        # Direct API call with shorter timeout
        url = "https://api.deepseek.com/v1/chat/completions"
        
        # Headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Payload
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": larger_prompt}],
            "temperature": 0,
            "max_tokens": 1000,  # Increased max tokens for the larger prompt
            "response_format": {"type": "json_object"}
        }
        
        # Make the API call with shorter explicit timeout
        logging.info("Making API request...")
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        
        elapsed = time.time() - start_time
        logging.info(f"API call completed in {elapsed:.2f} seconds with status {response.status_code}")
        
        # Check for success
        if response.status_code != 200:
            logging.error(f"API call failed with status {response.status_code}: {response.text}")
            return
        
        # Parse the response
        response_data = response.json()
        logging.info(f"Response: {json.dumps(response_data, indent=2)[:500]}...")
        
        # Extract content from the response
        if "choices" in response_data and len(response_data["choices"]) > 0:
            content = response_data["choices"][0].get("message", {}).get("content", "")
            
            # Log the response
            logging.info(f"Raw API response content: {content}")
            
            # Try to parse as JSON
            try:
                extracted_data = json.loads(content)
                logging.info(f"Successfully parsed JSON response: {json.dumps(extracted_data, indent=2)}")
                
                # Check for invoices array
                if "invoices" in extracted_data:
                    logging.info(f"Found 'invoices' array with {len(extracted_data['invoices'])} items")
                
            except json.JSONDecodeError as e:
                logging.error(f"Couldn't parse response as JSON: {e}")
        
    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        logging.error(f"API request timed out after {elapsed:.2f} seconds")
    except requests.RequestException as e:
        elapsed = time.time() - start_time
        logging.error(f"API request failed after {elapsed:.2f} seconds: {e}")
    except Exception as e:
        logging.error(f"Error during API test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    minimal_api_test()
