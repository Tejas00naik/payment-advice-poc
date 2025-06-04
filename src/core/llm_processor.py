#!/usr/bin/env python3
"""
LLM Normalization Engine for Payment Advice Extraction System.
Converts raw extracted data to structured format using OpenAI.
"""
import json
import os
import logging
import time
import uuid
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Union
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
    # Explicitly use OpenAI provider with gpt-4.1-mini model
    llm = create_llm(
        provider="openai",  # Explicitly use OpenAI
        model_name="gpt-4.1-mini",  # Explicitly use gpt-4.1-mini model
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
    Extracts global metadata and financial entries from raw text using an LLM.

    Args:
        text_data: Raw text data from PDF extraction.

    Returns:
        A dictionary containing 'global_metadata' (dict) and 'financial_entries' (list of dicts),
        or an empty dictionary if extraction fails.
    """
    try:
        # Get LLM with JSON output format and sufficient max_tokens
        llm = get_llm(temperature=0, json_mode=True, max_tokens=4000)
        if not llm:
            logging.error("Failed to initialize LLM")
            return {}
        
        # Define the prompt template for table extraction
        table_prompt = """
You are a payment advice processing specialist. Extract GLOBAL METADATA and ALL FINANCIAL ENTRIES from this document.

DOCUMENT TEXT:{text}

CRITICAL INSTRUCTIONS:
1. EXTRACT GLOBAL METADATA (from non-table text):
   - Payment Advice number (search: "Payment Advice No.", "Advice Number")
   - Sendor mail (look in headers/footers)
   - Original sendor mail (if different from Sendor mail)
   - Vendor name (search: "Vendor:", "Supplier:", "Payee Name")
   - Customer Name (search: "Customer:", "Buyer Name")

2. EXTRACT ALL FINANCIAL ENTRIES (from ALL tables):
   - Include every row representing: Invoices, Payments, BD/Coop Settlements, Debit Notes, TDS
   - Preserve original column names and values exactly
   - Include entries spanning multiple pages/sections
   - DO NOT skip any rows or combine entries
   - MAINTAIN ORIGINAL SIGN CONVENTION: 
        • Negative amounts = Credit to vendor account (our money)
        • Positive amounts = Debit to vendor account (their money)

OUTPUT FORMAT (STRICT JSON):
{{
  "global_metadata": {{
    "payment_advice_number": "value",
    "sendor_mail": "value",
    "original_sendor_mail": "value",
    "vendor_name": "value",
    "customer_name": "value"
  }},
  "financial_entries": [
    {{ "Column1": "Value1", "Column2": "Value2", ... }},  // Entry 1
    {{ "Column1": "Value1", "Column2": "Value2", ... }}   // Entry 2
  ]
}}

RULES:
- Use null for missing metadata fields
- Never add text outside JSON
- Preserve original table column names like "Invoice Ref", "TDS Amt"
- Include 20-50+ entries common in payment advices
- MAINTAIN ORIGINAL AMOUNT SIGNS (DO NOT convert signs)
"""
        
        # Extract all table-related text from the document
        extracted_text = text_data
        logging.info(f"Original text is {len(text_data)} chars")

        # First, check if we have page markers that might indicate multiple tables
        has_page_markers = any(marker in text_data for marker in ["--- PAGE ", "Page ", "page "])
        
        # Look for invoice-related markers
        invoice_markers = ["Invoice Number", "Invoice #", "Invoice ID", "Invoice Date", "Amount Paid", "Remaining"]
        
        # If the text is very long, we need to be smarter about extraction
        if len(text_data) > 3000:  
            logging.info("Text is long, extracting all table-related sections")
            
            # Find all potential table sections
            start_positions = []
            for marker in invoice_markers:
                pos = 0
                while True:
                    pos = text_data.find(marker, pos)
                    if pos == -1:
                        break
                    start_positions.append(pos)
                    pos += len(marker)
            
            if start_positions:
                # Sort positions to maintain order
                start_positions.sort()
                
                # If we have multiple positions, take the first one and extract enough text
                # to cover all potential table content
                start_idx = start_positions[0]
                # Use a much larger chunk to ensure we get all table data
                # Make this larger if we detect page markers which suggest multiple tables
                max_length = 5000 if has_page_markers else 3000
                end_idx = min(start_idx + max_length, len(text_data))
                extracted_text = text_data[start_idx:end_idx].strip()
                logging.info(f"Extracted table section from position {start_idx} ({len(extracted_text)} chars)")
            else:
                # If no invoice marker found, take the middle chunk which often contains tables
                middle = len(text_data) // 2
                extracted_text = text_data[max(0, middle - 1500):min(middle + 1500, len(text_data))].strip()
                logging.info(f"No specific markers found, using middle section ({len(extracted_text)} chars)")
        
        # Fill in the prompt template
        filled_prompt = table_prompt.replace("{text}", extracted_text)
        
        # Log timing details
        logging.info(f"Sending table extraction prompt to LLM (length: {len(filled_prompt)} chars)")
        start_time = time.time()
        
        try:
            # Use the direct API call method
            messages = [HumanMessage(content=filled_prompt)]
            logging.info("Making API call to LLM provider...")
            response = llm.invoke(messages)
            
            elapsed = time.time() - start_time
            logging.info(f"API call completed successfully in {elapsed:.2f} seconds")
            
            # Try to parse as JSON
            try:
                # The LLM is expected to return a JSON object with 'global_metadata' and 'financial_entries'
                extracted_data = json.loads(response.content)
                
                # Validate the structure
                if not isinstance(extracted_data, dict) or \
                   not all(k in extracted_data for k in ['global_metadata', 'financial_entries']) or \
                   not isinstance(extracted_data.get('global_metadata'), dict) or \
                   not isinstance(extracted_data.get('financial_entries'), list):
                    logging.error(f"LLM response does not match expected structure. Got: {list(extracted_data.keys()) if isinstance(extracted_data, dict) else type(extracted_data)}")
                    # Attempt to find 'financial_entries' if the top level is a list (old format recovery)
                    if isinstance(extracted_data, list):
                        logging.warning("LLM returned a list, attempting to use as financial_entries with empty global_metadata.")
                        return {"global_metadata": {}, "financial_entries": extracted_data}
                    return {}

                num_entries = len(extracted_data.get('financial_entries', []))
                logging.info(f"Successfully extracted global metadata and {num_entries} financial entries.")
                return extracted_data
            except json.JSONDecodeError:
                logging.error("Couldn't parse LLM response as JSON. Raw content:")
                logging.error(response.content[:500] + "..." if len(response.content) > 500 else response.content)
                
                # Fallback: try to extract JSON from markdown if present
                if "```json" in response.content:
                    try:
                        json_text = response.content.split("```json")[1].split("```")[0].strip()
                        extracted_data_md = json.loads(json_text)
                        if not isinstance(extracted_data_md, dict) or \
                           not all(k in extracted_data_md for k in ['global_metadata', 'financial_entries']) or \
                           not isinstance(extracted_data_md.get('global_metadata'), dict) or \
                           not isinstance(extracted_data_md.get('financial_entries'), list):
                            logging.error(f"Markdown JSON response does not match expected structure. Got: {list(extracted_data_md.keys()) if isinstance(extracted_data_md, dict) else type(extracted_data_md)}")
                            if isinstance(extracted_data_md, list):
                                logging.warning("Markdown LLM returned a list, attempting to use as financial_entries with empty global_metadata.")
                                return {"global_metadata": {}, "financial_entries": extracted_data_md}
                            return {}
                        
                        num_entries_md = len(extracted_data_md.get('financial_entries', []))
                        logging.info(f"Extracted global metadata and {num_entries_md} financial entries from markdown.")
                        return extracted_data_md
                    except (IndexError, json.JSONDecodeError) as e_md:
                        logging.error(f"Failed to extract JSON from markdown: {e_md}")
                
                return {}
                
        except Exception as e_api:
            logging.error(f"Error during table extraction API call: {str(e_api)}")
            return {}
                
    except Exception as e_main:
        logging.error(f"Table extraction error: {e_main}")
        import traceback
        traceback.print_exc()
        return {}


def normalize_extracted_data(extracted_data: dict):
    # Initialize tracking variables
    batch_size = 10  # Process 10 items at a time
    all_normalized_items = []
    
    # Create batch logging directory
    batch_dir = Path("results/llm_batches")
    batch_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created batch logging directory at {batch_dir}")
    """
    Normalize and structure financial entries using a pre-trained language model.
    It uses global metadata and financial entries extracted in a previous step.
    
    Implementation uses batch processing to handle large datasets without timeouts.
    
    Args:
        extracted_data: A dictionary containing:
            - 'global_metadata': A dictionary of global document metadata.
            - 'financial_entries': A list of dictionaries, where each is a financial entry (row).
        
    Returns:
        A normalized list of invoice items with consistent structure, or an empty list if issues occur.
    """
    # Get LLM for normalization
    llm = get_llm(temperature=0, json_mode=True, max_tokens=2000)
    if not llm:
        logging.error("Failed to initialize LLM for normalization")
        return []
    
    global_metadata = extracted_data.get('global_metadata', {})
    financial_entries = extracted_data.get('financial_entries', [])

    if not isinstance(global_metadata, dict):
        logging.error(f"'global_metadata' is not a dictionary. Found: {type(global_metadata)}. Using empty metadata.")
        global_metadata = {}
    if not isinstance(financial_entries, list):
        logging.error(f"'financial_entries' is not a list. Found: {type(financial_entries)}. Cannot process.")
        return []

    logging.info(f"Received {len(financial_entries)} financial entries to normalize.")
    logging.info(f"Global metadata to be used: {list(global_metadata.keys())}")

    # Process data in batches to avoid timeouts
    logging.info(f"Starting batch processing of {len(financial_entries)} entries")
    if not financial_entries:
        logging.warning("No financial entries to normalize.")
        return []
    
    # Create batch logging directory
    batch_dir = Path("results/llm_batches")
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    if not financial_entries:
        logging.warning("No financial entries to normalize.")
        return []
    
    table_data = financial_entries
    for i in range(0, len(table_data), batch_size):
        batch_data = table_data[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        batch_id = uuid.uuid4().hex[:8]
        
        logging.info(f"Processing batch {batch_num}: items {i+1}-{min(i+batch_size, len(table_data))} of {len(table_data)}")
        
        # Generate unique batch ID for tracking
        batch_id = str(uuid.uuid4())[:8]
        batch_input_file = batch_dir / f"batch_{batch_id}_input.json"
        batch_output_file = batch_dir / f"batch_{batch_id}_output.json"
        
        # Save input batch for debugging
        with open(batch_input_file, 'w') as f:
            json.dump({
                "global_metadata": global_metadata,
                "batch_data": batch_data
            }, f, indent=2)
        logging.info(f"Saved batch input to {batch_input_file}")
        
        # Process batch
        batch_results = process_batch(batch_data, global_metadata, llm)
        
        # Save output batch for debugging
        with open(batch_output_file, 'w') as f:
            json.dump(batch_results, f, indent=2)
        
        batch_id = str(uuid.uuid4())[:8]
        logging.info(f"Batch {batch_id}: Processed {len(batch_data)} entries, got {len(batch_results)} results")
        
        # Verify the expected number of entries - allow for payment entry
        allowed_counts = [len(batch_data), len(batch_data) + 1]
        if len(batch_results) not in allowed_counts:
            logging.warning(f"Batch {batch_id}: Entry count mismatch! Expected {len(batch_data)} or {len(batch_data)+1}, got {len(batch_results)}")
            
            # Save batch files for later debugging
            batch_dir = Path("results/llm_batches")
            batch_dir.mkdir(parents=True, exist_ok=True)
            
            batch_input_file = batch_dir / f"batch_{batch_id}_input.json"
            batch_output_file = batch_dir / f"batch_{batch_id}_output.json"
            
        # Log the number of items processed
        if len(batch_results) == len(batch_data) + 1:
            logging.info(f"Successfully normalized {len(batch_data)} items plus 1 payment entry (total: {len(batch_results)} items)")
        else:
            logging.info(f"Successfully normalized {len(batch_results)} items out of {len(batch_data)} expected items")
        
        # Verify batch results
        if batch_results and all(isinstance(item, dict) for item in batch_results):
            for item_idx, item in enumerate(batch_results):
                # Check for presence of key fields
                has_document_number = ('Invoice number' in item and item['Invoice number'] is not None and str(item['Invoice number']).strip() != "") or ('Other document number' in item and item['Other document number'] is not None and str(item['Other document number']).strip() != "")
                has_amount_settled = 'Amount settled' in item and item['Amount settled'] is not None
                has_entry_type = 'Entry type' in item and item['Entry type'] is not None and str(item['Entry type']).strip() != ""
                logging.debug(f"  Item {item_idx} in batch: Document number present: {has_document_number}, Amount settled present: {has_amount_settled}, Entry type present: {has_entry_type}")
                logging.debug(f"  Item {item_idx} content: {json.dumps(item)}")

        # Add batch results to final list
        if batch_results:
            all_normalized_items.extend(batch_results)
            
    logging.info(f"Successfully normalized {len(all_normalized_items)} items in total")
    return all_normalized_items

def process_batch(batch_data, global_metadata, llm):
    """
    Process a batch of table data rows for normalization.
    """
    # Log batch size and contents
    entry_count = len(batch_data)
    logging.info(f"Processing batch with {entry_count} financial entries")
    logging.debug(f"First entry in batch: {json.dumps(batch_data[0], indent=2) if batch_data else 'None'}")
    logging.debug(f"Global metadata keys: {list(global_metadata.keys())}")
    
    normalize_prompt = """
You are a payment advice normalization engine. Your task is to process financial entries and output standardized JSON objects.

CRITICAL REQUIREMENTS:
1. Process ALL {entry_count} input entries - do not skip any
2. Ensure each input entry has a corresponding output entry in the same order
3. Format all entries according to the 9-field schema below
4. Handle 'bdpo' entries correctly - they must be 'BDPO dr' type with document in 'Other document number'
5. IF payment information exists in metadata, add ONE EXTRA entry for it

INPUT:
- Financial entries: {batch_data}
- Metadata: {global_metadata}

ENTRY PROCESSING RULES:
1. Entry Types:
   - "BDPO dr": For ANY entry containing 'bdpo', 'co-op', 'coop', 'BD-Coop', or 'marketing' (CRITICAL: Check both invoice number AND description fields)
   - "TDS dr": For tax deduction entries (typically with -TDS- in the document number)
   - "Debit note dr": For debits, returns, damage
   - "Bank receipt dr": For payment entries (from metadata only)
   - "Invoice cr": For all remaining invoice entries

2. Document Fields:
   - Invoice entries → Use Invoice number field (Other document = null)
   - All other types → Use Other document field (Invoice number = null)

3. Payment Entry (Create ONE if metadata has payment/UTR info):
   - Entry type: 'Bank receipt dr'
   - Amount: Positive value from metadata
   - Other document number: Payment/UTR number
   - Other fields from metadata

OUTPUT SCHEMA (for EACH entry):
{
  "Payment Advice number": "[From metadata]",
  "Sendor mail": "[From metadata]",
  "Original sendor mail": "[From metadata]",
  "Vendor name (Payee name)": "[From metadata]",
  "Customer Name as per Payment advice": "[From metadata]",
  "Entry type": "[One of the 5 types]",
  "Amount settled": number,
  "Other document number": "[For non-invoice entries]",
  "Invoice number": "[Only for Invoice entries]"
}

Response Format: JSON array containing {entry_count} objects (plus optional payment entry)

FINAL CHECK:
- Array length should be {entry_count} or {entry_count}+1 if payment entry was added
- Each object must have all 9 fields (null for missing values)
- JSON must be valid
"""
    
    # Fill in prompt template
    filled_prompt = normalize_prompt.replace("{global_metadata}", json.dumps(global_metadata, indent=2))
    filled_prompt = filled_prompt.replace("{batch_data}", json.dumps(batch_data, indent=2))
    filled_prompt = filled_prompt.replace("{entry_count}", str(entry_count))
    
    messages = [HumanMessage(content=filled_prompt)]
    
    start_time = time.time()
    try:
        logging.info(f"Sending batch to LLM (entries: {entry_count}, prompt length: {len(filled_prompt)})")
        response = llm.invoke(messages)
        api_time = time.time() - start_time
        logging.info(f"LLM processing time: {api_time:.2f} seconds")
        
        # Log truncated response
        response_preview = response.content[:500] + "..." if len(response.content) > 500 else response.content
        logging.debug(f"LLM response preview: {response_preview}")
        
        # Parse the response
        try:
            # Save the raw response for debugging
            debug_path = f"debug_llm_response_{time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(debug_path, 'w') as f:
                f.write(response.content)
            logging.info(f"Saved raw LLM response to {debug_path}")
            
            parsed_llm_output = json.loads(response.content)
            logging.debug(f"Parsed LLM output type: {type(parsed_llm_output).__name__}")
            
            # Attempt to extract the list of items, accommodating various wrapper structures
            if isinstance(parsed_llm_output, list):
                normalized_batch = parsed_llm_output
                logging.info(f"LLM returned a list with {len(normalized_batch)} items")
            elif isinstance(parsed_llm_output, dict):
                # Check for common wrapper keys that might contain the list of items
                potential_wrapper_keys = ['result', 'results', 'items', 'data', 'entries', 'normalized_items', 'normalized_entries']
                extracted_list = False
                
                for key in potential_wrapper_keys:
                    if key in parsed_llm_output and isinstance(parsed_llm_output[key], list):
                        normalized_batch = parsed_llm_output[key]
                        extracted_list = True
                        logging.info(f"Extracted {len(normalized_batch)} items from '{key}' wrapper")
                        break
                
                if not extracted_list:
                    # If it's a dictionary but doesn't contain a recognized list under a common key,
                    # assume the dictionary itself is a single item
                    normalized_batch = [parsed_llm_output]
                    logging.warning("LLM returned a single object instead of an array, converting to list")
            
            # Ensure we have a list
            if isinstance(normalized_batch, dict):
                normalized_batch = [normalized_batch]
            
            # Verify entry count - now allowing for an additional payment entry
            allowed_counts = [entry_count, entry_count + 1]  # Allow exact count or +1 for payment entry
            if len(normalized_batch) not in allowed_counts:
                logging.warning(f"Expected {entry_count} or {entry_count+1} entries but received {len(normalized_batch)} entries from LLM")
                
                # Check if the last entry is a payment entry
                has_payment_entry = False
                if len(normalized_batch) > 0:
                    last_entry = normalized_batch[-1]
                    if isinstance(last_entry, dict) and last_entry.get('Entry type') == 'Bank receipt dr':
                        logging.info("Found payment entry as the last item")
                        has_payment_entry = True
            
                if not has_payment_entry and len(normalized_batch) != entry_count:
                    logging.warning(f"Entry count issue: Expected {entry_count} entries or {entry_count+1} with payment, got {len(normalized_batch)}")
                        
                # Log warning only if we still don't have the right count after unwrapping
                if len(normalized_batch) not in allowed_counts:
                    logging.warning(f"After unwrapping, still have count mismatch. Got: {len(normalized_batch)}")
                        
                debug_file = f"debug_llm_response_{time.strftime('%Y%m%d_%H%M%S')}.json"
                with open(debug_file, 'w') as f:
                    json.dump({
                        "prompt": filled_prompt,
                        "response": response.content,
                        "expected_count": entry_count,
                        "received_count": len(normalized_batch)
                    }, f, indent=2)
                logging.warning(f"Saved debug info to {debug_file}")
            
            return normalized_batch
            
        except json.JSONDecodeError as je:
            logging.error(f"Failed to parse LLM response: {str(je)}")
            logging.error(f"Raw response: {response.content}")
            return []
            
    except Exception as e:
        logging.error(f"Batch processing error: {str(e)}")
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
        normalized_items = normalize_extracted_data(table_data)
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
