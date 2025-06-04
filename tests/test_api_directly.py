#!/usr/bin/env python3
"""
Simple script to test the DeepSeek API directly with the extracted text
"""

import os
import json
import time
import logging
from dotenv import load_dotenv
from langchain.schema import HumanMessage
from langchain.chat_models import ChatOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

def get_test_llm():
    """Configure the DeepSeek API client with minimal settings"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
    
    logging.info("Initializing DeepSeek API with minimal config")
    return ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=api_key,
        openai_api_base="https://api.deepseek.com/v1",
        temperature=0,
        request_timeout=60.0,
        max_retries=2
    )

def test_extraction_prompt():
    """Test the extraction prompt with the extracted text"""
    # Read the extracted text
    try:
        with open("extracted_text.txt", "r") as f:
            text_data = f.read()
            logging.info(f"Read {len(text_data)} characters from extracted_text.txt")
    except Exception as e:
        logging.error(f"Error reading extracted_text.txt: {e}")
        return
    
    # Create the simple extraction prompt
    prompt = f"""
    You are a data extraction specialist. Extract the table from this payment advice document as JSON.
    
    Document text:
    ```
    {text_data}
    ```
    
    Please extract all invoice details as a JSON array of objects, where each object represents one invoice row.
    Keep the original column names as properties.
    """
    
    # Initialize LLM
    llm = get_test_llm()
    
    # Make the API call with basic timing
    logging.info(f"Sending prompt to DeepSeek API (length: {len(prompt)} chars)")
    start_time = time.time()
    
    try:
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        
        elapsed = time.time() - start_time
        logging.info(f"API call completed in {elapsed:.2f} seconds")
        
        # Print the raw response content
        print("\nRaw API Response Content:")
        print("------------------------")
        print(response.content)
        print("------------------------")
        
        # Try to parse as JSON
        try:
            json_data = json.loads(response.content)
            print("\nParsed JSON Response:")
            print(json.dumps(json_data, indent=2))
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse response as JSON: {e}")
    
    except Exception as e:
        logging.error(f"API call failed: {e}")

if __name__ == "__main__":
    test_extraction_prompt()
