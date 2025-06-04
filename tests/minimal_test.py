#!/usr/bin/env python3
"""
Minimal test for DeepSeek API
"""

import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_deepseek_minimal():
    # Get API key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY not found in environment variables")
        return
        
    # API endpoint
    url = "https://api.deepseek.com/v1/chat/completions"
    
    # Headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Simple payload with a basic question
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "Return a JSON array with 3 sample invoice objects"}],
        "temperature": 0,
        "max_tokens": 500,
        "response_format": {"type": "json_object"}
    }
    
    print("Sending minimal request to DeepSeek API...")
    
    try:
        # Make the API call with a short timeout
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0].get("message", {}).get("content", "")
                
                print("\nResponse content:")
                print(content)
                
                try:
                    json_data = json.loads(content)
                    print("\nParsed JSON:")
                    print(json.dumps(json_data, indent=2))
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
    test_deepseek_minimal()
