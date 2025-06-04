#!/usr/bin/env python3
"""
Direct extraction implementation for DeepSeek API
"""

import os
import json
import time
import logging
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def extract_invoice_data_direct(text_data):
    """
    Extract invoice data using direct DeepSeek API calls
    
    Args:
        text_data: The text to extract invoices from
        
    Returns:
        A list of dictionaries containing the extracted rows
    """
    # Get API key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        logging.error("DEEPSEEK_API_KEY not found in environment variables")
        return []
    
    # Inform about the extraction attempt
    logging.info(f"Attempting to extract invoice data directly ({len(text_data)} chars)")
    
    # Extract just the invoice section to make the API call more efficient
    invoice_section = text_data
    if "Invoice Number:" in text_data:
        start_idx = text_data.find("Invoice Number:")
        end_idx = len(text_data)
        
        # Look for a good end marker
        for marker in ["https://mail.google.com", "CONFIDENTIAL", "Email Footer"]:
            if marker in text_data[start_idx:]:
                end_pos = text_data.find(marker, start_idx)
                if end_pos > start_idx and end_pos < end_idx:
                    end_idx = end_pos
        
        # Extract just the invoice section
        invoice_section = text_data[start_idx:end_idx].strip()
        logging.info(f"Extracted invoice section ({len(invoice_section)} chars)")
    
    # Use simplified prompt for better API performance
    simplified_prompt = f"""
    Extract the invoice details from this payment advice text:
    
    ```
    {invoice_section}
    ```
    
    Return a JSON object with an 'invoices' array containing all invoice rows.
    Each invoice should have properties for invoice_number, date, description, and amount.
    """
    
    logging.info(f"Sending simplified extraction prompt to DeepSeek API (length: {len(simplified_prompt)} chars)")
    start_time = time.time()
    
    try:
        # Direct API call
        url = "https://api.deepseek.com/v1/chat/completions"
        
        # Headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Payload
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": simplified_prompt}],
            "temperature": 0,
            "max_tokens": 2000,
            "response_format": {"type": "json_object"}
        }
        
        # Make the API call with explicit timeout
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        elapsed = time.time() - start_time
        logging.info(f"API call completed in {elapsed:.2f} seconds with status {response.status_code}")
        
        # Check for success
        if response.status_code != 200:
            logging.error(f"API call failed with status {response.status_code}: {response.text}")
            return []
        
        # Parse the response
        response_data = response.json()
        
        # Extract content from the response
        if "choices" in response_data and len(response_data["choices"]) > 0:
            content = response_data["choices"][0].get("message", {}).get("content", "")
            
            # Log a snippet of the response
            content_snippet = content[:200] + "..." if len(content) > 200 else content
            logging.info(f"Raw API response content snippet: {content_snippet}")
            
            # Try to parse as JSON
            try:
                json_data = json.loads(content)
                
                # Check for invoices array
                if "invoices" in json_data and isinstance(json_data["invoices"], list):
                    invoices = json_data["invoices"]
                    logging.info(f"Successfully extracted {len(invoices)} invoices")
                    return invoices
                else:
                    logging.warning("No 'invoices' array found in JSON response")
                    
                    # Try to find any array in the response as fallback
                    for key, value in json_data.items():
                        if isinstance(value, list) and len(value) > 0:
                            logging.info(f"Found array under key '{key}' with {len(value)} items")
                            return value
                    
                    # Return the whole object as last resort
                    return [json_data]
            
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse JSON response: {e}")
                logging.debug(f"Raw content: {content}")
                return []
        else:
            logging.error("No choices in API response")
            return []
    
    except requests.RequestException as e:
        elapsed = time.time() - start_time
        logging.error(f"API request failed after {elapsed:.2f} seconds: {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error in extraction: {e}")
        return []

if __name__ == "__main__":
    # Test the function with extracted_text.txt
    try:
        with open("extracted_text.txt", "r") as f:
            text_data = f.read()
        
        invoices = extract_invoice_data_direct(text_data)
        print(f"Extracted {len(invoices)} invoices:")
        for i, invoice in enumerate(invoices[:5]):
            print(f"{i+1}. {invoice}")
        
        if len(invoices) > 5:
            print(f"... and {len(invoices)-5} more")
    except Exception as e:
        print(f"Error: {e}")
