#!/usr/bin/env python3
"""
LLM Normalization Engine for Payment Advice Extraction System.
Converts raw extracted data to structured format using OpenAI.
"""
import json
import os
import pandas as pd
from typing import List, Dict, Any, Union
import logging
from dotenv import load_dotenv

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
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
def get_llm(temperature=0.7, json_mode=True):
    """
    Get a singleton instance of the LLM client
    
    Args:
        temperature: The sampling temperature to use
        json_mode: Whether to force JSON output format
        
    Returns:
        A configured ChatOpenAI instance pointing to DeepSeek's API
    """
    # Use provided DeepSeek API key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    # Check if the DeepSeek API actually supports the response_format parameter
    # Some API providers don't support all OpenAI parameters
    response_format = None
    if json_mode:
        # Log that we're attempting to use JSON mode
        logging.info("Requesting JSON format from DeepSeek API")
        # Try with JSON mode, but be prepared for it not to work
        response_format = {"type": "json_object"}
    
    # Using OpenAI's wrapper but with base_url set to DeepSeek's API
    return ChatOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",  # DeepSeek API endpoint
        model="deepseek-chat",  # Using standard deepseek-chat model (non-r1)
        temperature=temperature,
        response_format=response_format,
        request_timeout=60.0,  # Add timeout to prevent hanging
        max_retries=2  # Limit retries to prevent excessive waiting
    )

# Define final schema columns
FINAL_COLUMNS = [
    "Payment date", "Payment advice number", "Invoice number",
    "Invoice date", "Customer name", "Customer id (SAP id)",
    "UTR", "Doc type", "Doc number", "Amount paid",
    "TDS deducted", "Currency", "PA link"
]


def extract_table_structure(text_data: str) -> Dict[str, Any]:
    """First step: Extract structured table data from raw text using LLM
    
    Args:
        text_data: Raw text data from PDF extraction
        
    Returns:
        Dictionary with payment details and structured table rows
    """
    # Define the prompt template for table extraction that explicitly asks for JSON
    table_extraction_prompt = ChatPromptTemplate.from_template("""
    You are a payment processing specialist. Extract structured table data from this payment advice document.
    
    Instructions:
    1. First, identify all payment metadata (payment number, date, currency, amount, customer name, etc.)
    2. Then, extract the invoice table data into rows with consistent columns
    3. Maintain the sign of amounts (negative for deductions, positive for payments)
    4. Format all dates as DD-MM-YYYY if possible
    5. Be comprehensive - extract ALL rows from the table
    
    Format your response as a valid JSON object with the following structure:
    {{
        "payment_details": {{
            "payment_number": "...",
            "payment_date": "...",
            "currency": "...",
            // other payment metadata
        }},
        "table_rows": [
            {{
                "invoice_number": "...",
                "invoice_date": "...",
                "amount": "...",
                // other invoice fields
            }},
            // more rows
        ]
    }}
    
    Document text:
    {text_data}
    """)
    
    # Get the LLM instance with JSON mode enabled
    llm = get_llm(temperature=0, json_mode=True)
    
    try:
        # Extract the table structure using direct LLM call with JSON output format
        response = llm.invoke(table_extraction_prompt.format(text_data=text_data))
        
        # Try to extract JSON from the response
        import re
        import json
        
        # Log the raw response for debugging
        logging.info(f"Raw LLM response content type: {type(response.content)}")
        
        # Extract JSON content
        content = response.content
        json_match = re.search(r'\{\s*"payment_details".*\}', content, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            try:
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error: {e}, extracted text: {json_str[:100]}...")
        
        # If structured JSON not found, try to parse the markdown table
        payment_details = {}
        table_rows = []
        
        # Extract payment details from response
        if "Payment Metadata" in content or "Payment Details" in content:
            metadata_lines = [line for line in content.split('\n') if ':' in line and not '|' in line]
            for line in metadata_lines:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip().lower().replace(' ', '_')
                    value = parts[1].strip()
                    payment_details[key] = value
        
        # Extract table rows using regex for markdown tables
        table_pattern = r'\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|'
        table_matches = re.findall(table_pattern, content)
        
        header_found = False
        headers = []
        
        for row in table_matches:
            row = [cell.strip() for cell in row]
            
            if not header_found and not any(h.lower() in ''.join(row).lower() for h in ['invoice', 'amount', 'date']):
                continue
            
            if not header_found:
                headers = [h.lower().replace(' ', '_') for h in row]
                header_found = True
                continue
            
            if all(cell == '' for cell in row):
                continue
                
            row_dict = {}
            for i, cell in enumerate(row):
                if i < len(headers):
                    row_dict[headers[i]] = cell
            
            if row_dict:
                table_rows.append(row_dict)
        
        return {
            "payment_details": payment_details,
            "table_rows": table_rows
        }
    
    except Exception as e:
        logging.error(f"Table extraction error: {e}")
        import traceback
        traceback.print_exc()
        # Return empty structure on error
        return {"payment_details": {}, "table_rows": []}


def normalize_extracted_data(extracted_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Second step: Normalize extracted table data into final structured format
    
    Args:
        extracted_data: Dictionary with payment details and table rows from first step
        
    Returns:
        List of normalized data items in final schema
    """
    # Import json at the function level to avoid reference errors
    import json
    import re
    
    # Check if we have data to normalize
    if not extracted_data or not extracted_data.get("table_rows"):
        logging.warning("No table rows to normalize")
        return []
    
    # Define the prompt template for normalization
    normalization_prompt = ChatPromptTemplate.from_template("""
    You are a payment processing specialist. Normalize this extracted payment data into a consistent format.
    
    Instructions:
    1. Convert each table row into a standardized item
    2. Output schema must include: {columns}
    3. Infer document types:
       - Contains 'TDS' → Doc Type = 'Deduction'
       - Contains 'RTV' → 'Return'
       - Contains 'FCN' → 'Chargeback'
       - Otherwise → 'Invoice'
    4. Convert all amounts to numeric values (remove commas/symbols)
    5. Format all dates as DD-MM-YYYY
    6. Apply payment metadata to all items
    
    Format your response as a valid JSON array with the following structure:
    [
        {{
            "Payment date": "DD-MM-YYYY",
            "Payment advice number": "...",
            "Invoice number": "...",
            "Invoice date": "DD-MM-YYYY",
            "Customer name": "...",
            "Customer id (SAP id)": "...",
            "UTR": "...",
            "Doc type": "Invoice/Deduction/Return/Chargeback",
            "Doc number": "...",
            "Amount paid": numeric_value,
            "TDS deducted": numeric_value,
            "Currency": "...",
            "PA link": "..."
        }},
        // more items
    ]
    
    Payment Details: {payment_details}
    
    Table Rows: {table_rows}
    """)
    
    # Get the LLM instance with JSON mode enabled
    llm = get_llm(temperature=0.1, json_mode=True)
    
    try:
        # Prepare the inputs
        payment_details_str = json.dumps(extracted_data["payment_details"])
        table_rows_str = json.dumps(extracted_data["table_rows"])
        columns_str = ", ".join(FINAL_COLUMNS)
        
        # Call the LLM directly with JSON output format
        response = llm.invoke(normalization_prompt.format(
            columns=columns_str,
            payment_details=payment_details_str,
            table_rows=table_rows_str
        ))
        
        # Try to extract JSON from the response
        # Log the raw response for debugging
        logging.info(f"Normalization response type: {type(response.content)}")
        
        # Try to find a JSON array in the content
        content = response.content
        json_match = re.search(r'\[\s*\{.*\}\s*\]', content, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            try:
                result = json.loads(json_str)
                if isinstance(result, list):
                    return result
                else:
                    logging.warning(f"Expected a list but got {type(result)}")
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error in normalization: {e}")
        
        # If we couldn't find a valid JSON array, try to create normalized items manually
        normalized_items = []
        payment_details = extracted_data.get("payment_details", {})
        
        # Common payment fields
        common_fields = {
            "Payment date": payment_details.get("payment_date", ""),
            "Payment advice number": payment_details.get("payment_number", ""),
            "Customer name": payment_details.get("customer_name", payment_details.get("payment_made_to", "")),
            "Currency": payment_details.get("currency", payment_details.get("payment_currency", ""))
        }
        
        # Process each table row
        for row in extracted_data.get("table_rows", []):
            item = common_fields.copy()  # Start with common fields
            
            # Map table row fields to normalized schema
            field_mapping = {
                "invoice_number": "Invoice number",
                "invoice_date": "Invoice date",
                "amount": "Amount paid",
                "amount_paid": "Amount paid"
            }
            
            for src_field, dest_field in field_mapping.items():
                if src_field in row:
                    item[dest_field] = row[src_field]
            
            # Infer document type
            doc_type = "Invoice"
            invoice_num = row.get("invoice_number", "") or ""
            invoice_desc = row.get("invoice_description", "") or ""
            
            if "TDS" in invoice_num or "TDS" in invoice_desc:
                doc_type = "Deduction"
            elif "RTV" in invoice_num or "RTV" in invoice_desc:
                doc_type = "Return"
            elif "FCN" in invoice_num or "FCN" in invoice_desc:
                doc_type = "Chargeback"
                
            item["Doc type"] = doc_type
            item["Doc number"] = invoice_num
            
            # Add missing fields with empty values
            for col in FINAL_COLUMNS:
                if col not in item:
                    item[col] = ""
            
            normalized_items.append(item)
        
        return normalized_items
            
    except Exception as e:
        logging.error(f"Normalization error: {e}")
        import traceback
        traceback.print_exc()
        return []


def normalize_with_llm(raw_data: Union[pd.DataFrame, str], metadata: Dict[str, str]) -> List[Dict[str, Any]]:
    """Two-step extraction process using LangChain:
    1. Extract table structure from text
    2. Normalize extracted data into final format
    
    Args:
        raw_data: Raw data from PDF extraction (text or DataFrame)
        metadata: Metadata extracted from PDF headers
        
    Returns:
        List of structured data items
    """
    # Convert DataFrame to text if necessary
    if isinstance(raw_data, pd.DataFrame):
        text_data = raw_data.to_string(index=False)
    else:
        text_data = str(raw_data)
    
    # Add metadata to the text data for context
    if metadata:
        text_data = f"METADATA:\n{json.dumps(metadata, indent=2)}\n\nCONTENT:\n{text_data}"
    
    # Step 1: Extract table structure
    logging.info("Step 1: Extracting table structure with LLM")
    extracted_data = extract_table_structure(text_data)
    
    # Merge extracted metadata with provided metadata
    combined_metadata = {**metadata, **extracted_data.get("payment_details", {})}
    
    # Step 2: Normalize the extracted data
    logging.info("Step 2: Normalizing extracted data with LLM")
    extracted_data["payment_details"] = combined_metadata
    normalized_items = normalize_extracted_data(extracted_data)
    
    return normalized_items


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
