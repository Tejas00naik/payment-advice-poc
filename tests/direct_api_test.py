#!/usr/bin/env python3
"""
Test DeepSeek API using direct requests instead of LangChain
"""

import os
import json
import time
import requests
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

def get_simplified_text():
    """Get a simplified version of the extracted text focusing just on the invoice table"""
    try:
        with open("extracted_text.txt", "r") as f:
            text_data = f.read()
            logging.info(f"Read {len(text_data)} characters from extracted_text.txt")

            # Extract just invoice data section to simplify
            if "Invoice Number:" in text_data and "Invoice description" in text_data:
                start_idx = text_data.find("Invoice Number:")
                reduced_text = text_data[start_idx:start_idx+1500]  # Just get a smaller chunk
                logging.info(f"Reduced text to {len(reduced_text)} characters")
                return reduced_text
            return text_data[:2000]  # Return first 2000 chars if we can't find the table
    except Exception as e:
        logging.error(f"Error reading extracted_text.txt: {e}")
        return "Error reading text file"

def call_deepseek_api_directly():
    """Call the DeepSeek API directly using requests"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        logging.error("DEEPSEEK_API_KEY not found in environment variables")
        return
    
    # Get simplified text
    text_data = get_simplified_text()
    
    # Create a simple prompt
    prompt = f"""
    Extract the invoice details table from this payment advice as a JSON array.
    
    Text:
    ```
    {text_data}
    ```
    
    Return ONLY a JSON array of invoice objects without any explanation.
    """
    
    # API endpoint
    url = "https://api.deepseek.com/v1/chat/completions"
    
    # Request headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Request payload
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "max_tokens": 2000
    }
    
    logging.info(f"Sending request to DeepSeek API with {len(prompt)} chars prompt")
    start_time = time.time()
    
    try:
        # Make the API call with a timeout
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        elapsed = time.time() - start_time
        logging.info(f"API call completed in {elapsed:.2f} seconds with status code {response.status_code}")
        
        # Check for success
        if response.status_code == 200:
            # Parse the JSON response
            response_data = response.json()
            
            # Extract the content from the response
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0].get("message", {}).get("content", "")
                
                print("\nRaw API Response Content:")
                print("------------------------")
                print(content)
                print("------------------------")
                
                # Try to parse as JSON
                try:
                    json_data = json.loads(content)
                    print("\nParsed JSON Response:")
                    print(json.dumps(json_data, indent=2))
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse response content as JSON: {e}")
            else:
                logging.error("No choices in API response")
                print(json.dumps(response_data, indent=2))
        else:
            logging.error(f"API call failed with status {response.status_code}")
            print(response.text)
    
    except requests.RequestException as e:
        elapsed = time.time() - start_time
        logging.error(f"Request failed after {elapsed:.2f} seconds: {e}")

if __name__ == "__main__":
    call_deepseek_api_directly()
