#!/usr/bin/env python3
"""
LLM Normalization Engine for Payment Advice Extraction System.
Converts raw extracted data to structured format using OpenAI.
"""
import json
import os
import time
import pandas as pd
from typing import List, Dict, Any, Union
import logging
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Import our new adaptable LLM provider
from src.core.llm_provider import create_llm, LLMProvider
from langchain.schema import HumanMessage
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Define the schema for table extraction
class TableRow(BaseModel):
    """Schema for a single row in an extracted table"""
    invoice_number: str = Field(description="Invoice or document number", default="")
    invoice_date: str = Field(description="Date of the invoice in DD-MM-YYYY format", default="")
    description: str = Field(description="Description of the invoice or payment", default="")
    amount: str = Field(description="Amount with sign (negative for deductions)", default="")
    amount_paid: str = Field(description="Amount paid", default="")
    currency: str = Field(description="Currency code (e.g., USD, EUR, INR)", default="")

class TableExtraction(BaseModel):
    """Schema for extracted table data"""
    payment_details: Dict[str, str] = Field(
        description="Key-value pairs of payment metadata like payment number, date, currency, etc."
    )
    table_rows: List[TableRow] = Field(
        description="List of rows extracted from the invoice table"
    )

# Define the schema for final normalized output
class NormalizedItem(BaseModel):
    """Schema for a normalized payment advice item"""
    invoice_number: str = Field(description="Invoice or document number")
    invoice_date: str = Field(description="Date of the invoice in DD-MM-YYYY format")
    amount_paid: float = Field(description="Amount paid (positive for payments, negative for deductions)")
    doc_type: str = Field(description="Document type (Invoice, Deduction, Return, Chargeback)")
    payment_date: str = Field(description="Payment date in DD-MM-YYYY format")
    payment_advice_number: str = Field(description="Payment advice or remittance number")
    currency: str = Field(description="Currency code (e.g., USD, EUR, INR)")
    customer_name: str = Field(description="Name of the customer or payment sender")

class NormalizedOutput(BaseModel):
    """Schema for the complete normalized output"""
    items: List[NormalizedItem] = Field(description="List of normalized payment advice items")

# Initialize LangChain components
def get_llm(temperature=0, json_mode=True, max_tokens=2000):
    """
    Get an instance of the LLM client using the adaptable LLM provider
    
    Args:
        temperature: The sampling temperature to use (default: 0 for deterministic output)
        json_mode: Whether to force JSON output format
        max_tokens: Maximum tokens to generate (preventing truncation of JSON)
        
    Returns:
        A configured LLM instance (ChatOpenAI) for the detected provider
    """
    # Use our adaptable LLM provider
    # It automatically selects the provider based on available API keys
    llm = create_llm(
        provider=None,  # Auto-select provider based on available API keys
        model_name=None,  # Use default model for selected provider
        temperature=temperature,
        json_mode=json_mode,
        max_tokens=max_tokens,
        request_timeout=60.0  # 60-second timeout for API requests
    )
    
    if not llm:
        logging.error("Failed to initialize LLM provider")
        return None
        
    return llm

# Define final schema columns
FINAL_COLUMNS = [
    "Payment date", "Payment advice number", "Invoice number",
    "Invoice date", "Customer name", "Customer id (SAP id)",
    "UTR", "Doc type", "Doc number", "Amount paid",
    "TDS deducted", "Currency", "PA link"
]


def extract_table_structure(text_data: str):
    """
    First step: Extract ONLY table structure from raw text
    
    Args:
        text_data: Raw text data from PDF extraction
        
    Returns:
        List of extracted table rows
    """
    try:
        # Get LLM with JSON output format and sufficient max_tokens
        llm = get_llm(temperature=0, json_mode=True, max_tokens=4000)
        if not llm:
            logging.error("Failed to initialize LLM")
            return []
        
        # Define the prompt template based on our successful tests
        # Using a more structured format with clear instructions
        table_prompt = """
        You are a payment processing specialist. Extract all table information from this payment advice as JSON.
        
        Text from the payment advice:
        ```
        {text}
        ```
        
        Instructions:
        1. Focus ONLY on extracting tables containing invoice or payment details
        2. Extract each row of the table as a separate item
        3. Ensure each item has consistent fields (for eg invoice_number, date, description, amount)
        
        Return JSON with an 'invoices' array in this example format(actual columns can be different):
        {{  
          "invoices": [
            {{  
              "invoice_number": "...",
              "date": "...",
              "description": "...", 
              "amount": "..."
            }},
            // more items
          ]
        }}
        
        EXAMPLE OUTPUT JSON:
        {{  
          "invoices": [
            {{  
              "invoice_number": "INV-123",
              "date": "01-04-2023",
              "description": "Product supplies", 
              "amount": "1250.00"
            }},
            {{  
              "invoice_number": "INV-124",
              "date": "05-04-2023",
              "description": "Service fee", 
              "amount": "499.99"
            }}
          ]
        }}
        
        Return only valid JSON with the invoices array and nothing else.
        """
        
        # Extract a smaller subset of the text to improve API response time
        extracted_text = text_data
        if len(text_data) > 2000:  # Only process if the text is relatively long
            logging.info(f"Original text is {len(text_data)} chars, extracting relevant section")
            # Look for invoice-related markers
            invoice_markers = ["Invoice Number", "Invoice #", "Invoice ID", "Invoice Date"]
            start_idx = -1
            for marker in invoice_markers:
                if marker in text_data:
                    start_idx = text_data.find(marker)
                    logging.info(f"Found marker '{marker}' at position {start_idx}")
                    break
            
            if start_idx >= 0:
                # Extract a reasonable chunk starting from the marker
                # At most 1500 characters which should be enough for several invoices
                max_length = 1500
                end_idx = min(start_idx + max_length, len(text_data))
                extracted_text = text_data[start_idx:end_idx].strip()
                logging.info(f"Extracted relevant section ({len(extracted_text)} chars)")
            else:
                # If no invoice marker found, take first 1000 chars
                extracted_text = text_data[:1000].strip()
                logging.info(f"No invoice markers found, using first {len(extracted_text)} chars")
        
        # Fill in the prompt template
        filled_prompt = table_prompt.replace("{text}", extracted_text)
        
        # Log timing details
        logging.info(f"Sending table extraction prompt to DeepSeek API (length: {len(filled_prompt)} chars)")
        start_time = time.time()
        
        try:
            # Use the direct API call method
            messages = [HumanMessage(content=filled_prompt)]
            logging.info("Making API call to DeepSeek...")
            response = llm.invoke(messages)
            
            elapsed = time.time() - start_time
            logging.info(f"API call completed successfully in {elapsed:.2f} seconds")
            
            # Try to parse as JSON
            try:
                table_data = json.loads(response.content)
                
                # Handle various nested response formats from DeepSeek API
                if isinstance(table_data, dict):
                    # Common wrapper keys we've observed
                    possible_keys = ['response', 'table_rows', 'items', 'data', 'result', 'tables']
                    
                    for key in possible_keys:
                        if key in table_data and isinstance(table_data[key], list):
                            table_data = table_data[key]
                            logging.info(f"Unwrapped response from nested JSON structure with key '{key}'")
                            break
                
                logging.info(f"Successfully extracted {len(table_data)} table rows")
                return table_data
            except json.JSONDecodeError:
                logging.error("Couldn't parse response as JSON. Raw content:")
                logging.error(response.content[:500] + "..." if len(response.content) > 500 else response.content)
                
                # If we failed, try to extract JSON from the response if it contains markdown formatting
                if "```json" in response.content:
                    try:
                        # Extract JSON from markdown code block
                        json_text = response.content.split("```json")[1].split("```")[0].strip()
                        table_data = json.loads(json_text)
                        logging.info(f"Extracted JSON from markdown: {len(table_data)} table rows")
                        return table_data
                    except (IndexError, json.JSONDecodeError) as e:
                        logging.error(f"Failed to extract JSON from markdown: {e}")
                
                return []
                
        except Exception as e:
            logging.error(f"Error during table extraction API call: {str(e)}")
            return []
                
    except Exception as e:
        logging.error(f"Table extraction error: {e}")
        import traceback
        traceback.print_exc()
        return []


def normalize_extracted_data(table_data, full_text: str):
    """
    Second step: Normalize extracted table data into final structured format
    
    Args:
        table_data: Extracted table rows from first step
        full_text: Full text of the PDF for context
    
    Returns:
        List of normalized items
    """
    try:
        # Get LLM with JSON output format and sufficient max_tokens
        llm = get_llm(temperature=0, json_mode=True, max_tokens=4000)
        if not llm:
            logging.error("Failed to initialize LLM")
            return []
        
        # Define the prompt template for normalization with explicit JSON formatting
        # Note: Including the word "JSON" explicitly in the prompt as recommended by DeepSeek
        normalize_prompt = """
        You are a payment processing specialist. Convert the extracted table data into a normalized format as JSON.
        
        The extracted table data is:
        {table_data}
        
        The full document text is:
        {full_text}
        
        Instructions:
        1. First, extract key metadata from the full document text:
           - Payment advice number/reference
           - Payment date
           - Currency
           - Customer name
           - Any other payment metadata
        
        2. For each row in the extracted table, create a normalized payment advice item
        3. Combine the table data with the metadata you extracted from the full text
        4. Ensure all dates are in DD-MM-YYYY format
        5. Ensure amount values are numeric (can be negative for deductions)
        
        You MUST format your response EXACTLY as a valid JSON array of normalized items with this structure:
        [
            {{
                "invoice_number": "...",
                "invoice_date": "...",
                "amount_paid": 123.45,
                "doc_type": "Invoice",
                "payment_date": "...",
                "payment_advice_number": "...",
                "currency": "...",
                "customer_name": "..."
            }},
            // more items
        ]
        
        EXAMPLE OUTPUT JSON:
        [
            {{
                "invoice_number": "INV-123",
                "invoice_date": "01-04-2023",
                "amount_paid": 1250.00,
                "doc_type": "Invoice", 
                "payment_date": "15-04-2023",
                "payment_advice_number": "PA-789",
                "currency": "USD",
                "customer_name": "ACME Corporation"
            }},
            {{
                "invoice_number": "INV-124",
                "invoice_date": "05-04-2023",
                "amount_paid": 499.99,
                "doc_type": "Invoice",
                "payment_date": "15-04-2023",
                "payment_advice_number": "PA-789",
                "currency": "USD",
                "customer_name": "ACME Corporation"
            }}
        ]
        
        Only return the JSON array and nothing else.
        """
        
        # Fill in the prompt template
        filled_prompt = normalize_prompt.replace("{table_data}", json.dumps(table_data, indent=2))
        filled_prompt = filled_prompt.replace("{full_text}", full_text)
        
        logging.info(f"Sending normalization prompt to DeepSeek API (length: {len(filled_prompt)} chars)")
        import time
        start_time = time.time()
        
        try:
            # Use the direct API call method
            messages = [HumanMessage(content=filled_prompt)]
            response = llm.invoke(messages)
            
            elapsed = time.time() - start_time
            logging.info(f"API call completed in {elapsed:.2f} seconds")
            
            # Try to parse as JSON
            try:
                normalized_data = json.loads(response.content)
                
                # Handle various nested response formats from DeepSeek API
                if isinstance(normalized_data, dict):
                    # Common wrapper keys we've observed
                    possible_keys = ['response', 'normalized_payments', 'items', 'data', 'result', 'normalized_items']
                    
                    for key in possible_keys:
                        if key in normalized_data and isinstance(normalized_data[key], list):
                            normalized_data = normalized_data[key]
                            logging.info(f"Unwrapped response from nested JSON structure with key '{key}'")
                            break
                
                logging.info(f"Successfully normalized {len(normalized_data)} items")
                return normalized_data
            except json.JSONDecodeError:
                logging.error("Couldn't parse response as JSON. Raw content:")
                logging.error(response.content[:500] + "..." if len(response.content) > 500 else response.content)
                
                # If we failed, try to extract JSON from the response if it contains markdown formatting
                if "```json" in response.content:
                    try:
                        # Extract JSON from markdown code block
                        json_text = response.content.split("```json")[1].split("```")[0].strip()
                        normalized_data = json.loads(json_text)
                        logging.info(f"Extracted JSON from markdown: {len(normalized_data)} normalized items")
                        return normalized_data
                    except (IndexError, json.JSONDecodeError) as e:
                        logging.error(f"Failed to extract JSON from markdown: {e}")
                
                return []
                
        except Exception as e:
            logging.error(f"Error during normalization API call: {str(e)}")
            return []
                
    except Exception as e:
        logging.error(f"Normalization error: {e}")
        import traceback
        traceback.print_exc()
        return []


def normalize_with_llm(raw_data: Union[pd.DataFrame, str], metadata: Dict[str, str] = None, save_json_output: bool = False) -> List[Dict[str, Any]]:
    """Two-step extraction process using direct API calls with JSON-optimized prompts
    1. Extract table structure from text
    2. Normalize extracted data into final format with full text context
    
    Args:
        raw_data: Raw data from PDF extraction (text or DataFrame)
        metadata: Optional metadata extracted from PDF headers
        save_json_output: Whether to save intermediate JSON outputs
        
    Returns:
        List of structured data items
    """
    try:
        # Convert DataFrame to text if necessary
        if isinstance(raw_data, pd.DataFrame):
            text_data = raw_data.to_string(index=False)
        else:
            text_data = str(raw_data)
        
        # Add metadata to the text data for context if provided
        if metadata and metadata:
            metadata_text = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
            text_data = f"METADATA:\n{metadata_text}\n\nCONTENT:\n{text_data}"
        
        # Log the beginning of the process with timestamp
        import time
        start_time = time.time()
        logging.info(f"Starting two-step extraction process on text of length {len(text_data)}")
        
        # Step 1: Extract table structure with improved JSON-optimized function
        table_data = extract_table_structure(text_data)
        if not table_data or len(table_data) == 0:
            logging.warning("No table data extracted, returning empty result")
            return []
            
        # Log completion of first step
        step1_time = time.time() - start_time
        logging.info(f"Step 1 (Table extraction) completed in {step1_time:.2f} seconds. Found {len(table_data)} rows.")
            
        # Optional: Save extracted table data to JSON
        if save_json_output:
            with open('test_table_data.json', 'w') as f:
                json.dump(table_data, f, indent=2)
                logging.info(f"Saved extracted table data to test_table_data.json")
                
        # Step 2: Normalize extracted data with full context using improved JSON-optimized function
        step2_start = time.time()
        normalized_items = normalize_extracted_data(table_data, text_data)
        if not normalized_items or len(normalized_items) == 0:
            logging.warning("No normalized data extracted, returning empty result")
            return []
            
        # Log completion of second step
        step2_time = time.time() - step2_start
        total_time = time.time() - start_time
        logging.info(f"Step 2 (Normalization) completed in {step2_time:.2f} seconds")
        logging.info(f"Total extraction process completed in {total_time:.2f} seconds")
            
        # Optional: Save normalized data to JSON
        if save_json_output:
            with open('test_normalized_data.json', 'w') as f:
                json.dump(normalized_items, f, indent=2)
                logging.info(f"Saved normalized data to test_normalized_data.json")
        
        return normalized_items
                
    except Exception as e:
        logging.error(f"Error in normalize_with_llm: {e}")
        import traceback
        traceback.print_exc()
        return []


def safe_llm_normalization(raw_data, metadata, max_retries=2):
    """
    Retry mechanism for LLM normalization with fallback to regex parsing
    
    Args:
        raw_data: Raw data from PDF extraction (text or DataFrame)
        metadata: Metadata extracted from PDF headers
        max_retries: Maximum number of retries before falling back to regex
        
    Returns:
        List of structured data items
    """
    retries = 0
    while retries <= max_retries:
        try:
            logging.info(f"LLM normalization attempt {retries+1}/{max_retries+1}")
            items = normalize_with_llm(raw_data, metadata)
            if items and len(items) > 0:
                return items
            else:
                logging.warning("LLM returned empty results, retrying...")
        except Exception as e:
            logging.error(f"LLM retry {retries+1}/{max_retries+1} failed: {e}")
        retries += 1
        
    # Fallback to regex parsing if LLM fails
    logging.warning("LLM processing failed after retries, falling back to regex parsing")
    return fallback_regex_parsing(raw_data, metadata)


def fallback_regex_parsing(raw_data, metadata):
    """
    Fallback parser using regex for when LLM fails
    Basic extraction of invoice numbers and amounts
    """
    import re
    
    result = []
    
    # Convert DataFrame to string if needed
    if isinstance(raw_data, pd.DataFrame):
        text = raw_data.to_string()
    else:
        text = str(raw_data)
    
    # Find invoice patterns
    invoice_matches = re.finditer(r'(?:INV|Invoice)[- ]?(?:No|Number)?[:\s]*([A-Z0-9/-]+)', text, re.IGNORECASE)
    
    for match in invoice_matches:
        invoice_number = match.group(1).strip()
        
        # Find amount near invoice (within next 200 chars)
        context = text[match.start():min(match.start() + 200, len(text))]
        amount_match = re.search(r'(?:Amount|Paid|Total)[:\s]*(?:Rs\.|₹|$|€|£)?[\s]*([0-9,.]+)', context, re.IGNORECASE)
        
        result.append({
            "Invoice number": invoice_number,
            "Amount paid": amount_match.group(1).replace(",", "") if amount_match else None,
            "Doc type": "Invoice"
        })
    
    return result
