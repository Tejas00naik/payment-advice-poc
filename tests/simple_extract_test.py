#!/usr/bin/env python3
"""
Simplified extraction test for DeepSeek API with real data
"""

import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_invoice_section():
    """Extract just the invoice table section from the text file"""
    try:
        with open("extracted_text.txt", "r") as f:
            full_text = f.read()
            
            # Find the start of the invoice section
            if "Invoice Number:" in full_text:
                start_idx = full_text.find("Invoice Number:")
                end_idx = full_text.find("https://mail.google.com", start_idx)
                
                if end_idx > start_idx:
                    # Extract just the invoice table section
                    invoice_section = full_text[start_idx:end_idx].strip()
                    print(f"Extracted invoice section ({len(invoice_section)} chars)")
                    return invoice_section
            
            # Fallback to just the first part if we can't find the section
            short_text = full_text[:1500]
            print(f"Using first {len(short_text)} chars of the document")
            return short_text
    except Exception as e:
        print(f"Error reading file: {e}")
        return "Error reading file"

def test_invoice_extraction():
    # Get API key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY not found in environment variables")
        return
    
    # Get invoice section
    invoice_text = get_invoice_section()
    
    # API endpoint
    url = "https://api.deepseek.com/v1/chat/completions"
    
    # Headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Simple prompt focusing just on invoice extraction
    prompt = f"""
    Extract the invoice details from this text:
    
    ```
    {invoice_text}
    ```
    
    Return a JSON array of invoice objects with properties for invoice number, date, description, and amount.
    """
    
    # Payload
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 1000,
        "response_format": {"type": "json_object"}
    }
    
    print(f"Sending request to DeepSeek API with prompt length: {len(prompt)}")
    
    try:
        # Make the API call with a timeout
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0].get("message", {}).get("content", "")
                
                print("\nResponse content:")
                print(content[:500] + "..." if len(content) > 500 else content)
                
                # Try to parse as JSON
                try:
                    json_data = json.loads(content)
                    print("\nSuccessfully parsed JSON response")
                    
                    # Save the response to a file
                    with open("deepseek_invoice_response.json", "w") as f:
                        json.dump(json_data, f, indent=2)
                    print("Saved full response to deepseek_invoice_response.json")
                    
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {e}")
            else:
                print("No choices in response")
                print(response_data)
        else:
            print(f"Error response: {response.text}")
    
    except requests.RequestException as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_invoice_extraction()
