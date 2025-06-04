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

# Maximum text length for extraction to avoid token limits
MAX_EXTRACTION_TEXT_LEN = 16000

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
You are a financial document analyzer specialized in extracting structured data from payment advice PDF documents.

MISSION: Extract ALL financial data from the payment advice PDF text in a structured JSON format

DOCUMENT TEXT: {text}

1. IDENTIFY GLOBAL METADATA (apply to the entire document):
   - Payment advice number/ID
   - Sender email address(es)
   - Vendor/payee name
   - Customer name
   - Payment date (search for dates near payment amount)
   - Payment reference/UTR number (IMPORTANT: Look for "Payment number", "Payment No.", "Payment Reference", "UTR", "Bank Ref" - this is the bank transaction reference)
   - Payment amount (EXTREMELY IMPORTANT: Look carefully for the TOTAL amount being paid, normally preceded by text like "Payment amount:", "Total:", "Net Amount:" or at the end of the document. Payment amounts are typically large numbers like "INR 3,500,000.00")

2. EXTRACT ALL FINANCIAL ENTRIES (from ALL tables):
   - CRITICAL: Extract EVERY SINGLE ROW from the tables. Do NOT skip any rows.
   - Count the number of rows in the tables and ensure that many entries are included
   - IMPORTANT: Use the ACTUAL column headers from the document
   - Look for tables with financial data (invoices, payments, deductions, etc.)
   - Extract ALL rows from ALL tables with their original column headers
   - Preserve the exact column names as they appear in the document
   - For amount fields: Use proper number format with signs (negative for debits/deductions)
   - Include every row representing: Invoices, Payments, Settlements, Debit Notes, TDS, etc.
   - If the table contains line numbers (Sr No., S.No, etc.), verify that you've extracted ALL rows by checking the sequence

OUTPUT FORMAT:
{
  "global_metadata": {
    "payment_advice_number": "string",
    "sendor_mail": "string",
    "original_sendor_mail": "string",
    "vendor_name": "string",
    "customer_name": "string",
    "payment_date": "string",
    "payment_amount": number,
    "payment_utr": "string",
    "payment_number": "string",
    "total_row_count": number  // Add this field with the total number of rows you identified
  },
  "financial_entries": [
    // Use the ACTUAL column headers from the document
    // Each entry should be a JSON object with keys matching the table headers
    // Example (but don't hardcode these field names, use what's in the document):
    // {
    //   "Sr No.": "1",
    //   "Type of Document": "Invoice",
    //   "Doc No": "INV123",
    //   "Amount": 1000.00,
    //   ... (all other columns from the actual document)
    // }
  ]
}

EXTREMELY IMPORTANT ABOUT PAYMENT AMOUNT:
- The payment amount is one of the most critical fields to extract
- It represents the TOTAL amount being transferred to the vendor
- In the sample document, the payment amount is approximately INR 3,511,844.52
- This amount typically appears at the end of the document or near text mentioning "payment"
- If you see multiple amounts, choose the largest positive amount that appears to be a total
- The payment amount should be a POSITIVE number (money being paid)

IMPORTANT INSTRUCTIONS:
- Be thorough in finding all payment information
- Include negative signs for deductions (like TDS)
- Extract ALL rows from ALL tables
- Format numbers as numbers (not strings) when possible
- Return valid JSON
"""
        
        # Extract all table-related text from the document
        document_text = text_data
        logging.info(f"Original text is {len(text_data)} chars")

        # First, check if we have page markers that might indicate multiple tables
        has_page_markers = any(marker in text_data for marker in ["--- PAGE ", "Page ", "page "])
        
        # We'll use a chunking approach to avoid token limits
        max_text_length = MAX_EXTRACTION_TEXT_LEN
        logging.info(f"Document text length: {len(document_text)} chars")
        
        # Initialize extracted_text with the document_text
        extracted_text = document_text
        
        # If text is too long, focus on main content (remove headers/footers but keep all tables)
        if len(document_text) > max_text_length:
            # Look for key indicators of the main content area
            table_start_markers = ["Invoice Number", "Document No", "Ref No", "Item", "Description", "Amount"]
            earliest_marker_pos = float('inf')
            
            # Find the earliest indicator of table content
            for marker in table_start_markers:
                pos = document_text.lower().find(marker.lower())
                if pos != -1 and pos < earliest_marker_pos:
                    earliest_marker_pos = pos
            
            # If we found a marker, start a bit before it
            if earliest_marker_pos < float('inf'):
                # Include some context before the marker
                start_pos = max(0, earliest_marker_pos - 300)  
                # Take as much content as possible within limits
                extracted_text = document_text[start_pos:start_pos + max_text_length]
                logging.info(f"Focused on main content area ({len(extracted_text)} chars)")
            else:
                # If no markers found, take the beginning of the document
                extracted_text = document_text[:max_text_length]
                logging.info(f"No table markers found, using document start ({len(extracted_text)} chars)")
        
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


def extract_table_sections(text):
    """Extract sections of text likely to contain tables or lists.
    
    Args:
        text (str): The document text
        
    Returns:
        str: Text sections likely containing tables
    """
    # Look for invoice-related markers to identify table sections
    invoice_markers = ["Invoice Number", "Invoice #", "Invoice Date", "Amount Paid", "Remaining"]
    marker_positions = []
    
    # Find positions of all table markers
    for marker in invoice_markers:
        pos = 0
        while True:
            pos = text.find(marker, pos)
            if pos == -1:
                break
            marker_positions.append(pos)
            pos += len(marker)
    
    if not marker_positions:
        # No markers found, return the entire text
        return text
    
    # Sort positions and extract the relevant sections
    marker_positions.sort()
    first_marker = marker_positions[0]
    # Extract from the first marker to the end, with some padding before
    start_pos = max(0, first_marker - 500)
    extracted_text = text[start_pos:]
    
    return extracted_text


def extract_document_structure(text, llm):
    """Extract structured data including tables from the document text.
    
    Args:
        text (str): The document text
        llm (LlmProvider): LLM provider instance
        
    Returns:
        dict: Structured data with global_metadata and financial_entries
    """
    # Use the table_extract function to get structured data
    table_data = table_extract(text, llm)
    
    # Verify we have the expected format and keys
    if not isinstance(table_data, dict):
        logging.warning(f"Unexpected table_data structure: {type(table_data).__name__}")
        return {"global_metadata": {}, "financial_entries": []}
    
    # Ensure we have the necessary keys
    if 'global_metadata' not in table_data:
        table_data['global_metadata'] = {}
    if 'financial_entries' not in table_data:
        table_data['financial_entries'] = []
    
    # Log the extracted data
    logging.info(f"Extracted {len(table_data.get('financial_entries', []))} financial entries")
    logging.info(f"Global metadata keys: {list(table_data.get('global_metadata', {}).keys())}")
    
    # Process BDPO entries
    for entry in table_data.get('financial_entries', []):
        # If an invoice description contains BDPO-related terms, mark it as a BDPO entry
        if isinstance(entry, dict) and 'Invoice description' in entry and entry['Invoice description']:
            desc = str(entry['Invoice description']).lower()
            if any(marker in desc for marker in ['bdpo', 'co-op', 'coop', 'bd-coop', 'marketing']):
                entry['is_bdpo'] = True
                
    # Save the extracted data for debugging
    with open(f"extracted_structure_debug_{time.strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(table_data, f, indent=2)
    
    return table_data


def normalize_extracted_data(extracted_data: dict):
    """Normalize and structure financial entries using a pre-trained language model.
    
    Args:
        extracted_data: A dictionary containing:
          - global_metadata (dict): Global metadata for the document
          - financial_entries (list): List of financial entries extracted from tables
          
    Returns:
        list: List of standardized and normalized financial entries
    """
    # Initialize tracking variables
    all_normalized_items = []
    
    # Track batches
    batch_idx = 0
    total_items_count = len(extracted_data.get('financial_entries', []))
    logging.info(f"Total entries to normalize: {total_items_count}")
    
    # Set a smaller batch size to ensure all rows are processed correctly without hitting token limits
    # For small sets, process all at once; for larger sets, use smaller batches
    if total_items_count <= 5:
        batch_size = total_items_count
    else:
        batch_size = 5  # Smaller batch size to avoid token limits and ensure all rows are processed
    
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
        
        # Process current batch (first batch flag is True only for the first batch)
        is_first_batch = (batch_num == 1)
        
        # Add retry logic for batches - try up to 2 times with increasing emphasis on preserving all rows
        max_retry = 2
        retry_count = 0
        batch_results = None
        
        while retry_count <= max_retry:
            # For retries, add special flag to emphasize preserving ALL rows
            retry_flag = retry_count > 0
            
            # Process the batch
            batch_results = process_batch(batch_data, global_metadata, llm, is_first_batch, retry_flag)
            
            # Check if we got all the expected rows
            if len(batch_results) >= len(batch_data):  # Allow for potential payment entry
                logging.info(f"Batch processing successful with {len(batch_results)} results from {len(batch_data)} entries")
                break
            else:
                retry_count += 1
                logging.warning(f"Missing rows in batch processing - retry attempt {retry_count}/{max_retry}")
                
                # If we've exceeded retries, use the last result we got
                if retry_count > max_retry:
                    logging.warning(f"Failed to get all rows after {max_retry} retries. Proceeding with {len(batch_results)} out of {len(batch_data)}")
                    break
        
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
            
    # Verify the total row count before returning results
    final_count = len(all_normalized_items)
    total_original_count = len(financial_entries)
    
    # Add a flag in the metadata to track row count validation
    if 'total_row_count' not in global_metadata:
        global_metadata['total_row_count'] = total_original_count
    
    # Validate the row counts (allowing for payment entry)
    if final_count < total_original_count:
        logging.warning(f"ROW COUNT MISMATCH: Expected at least {total_original_count} normalized entries but got only {final_count}")
        
        # Detailed analysis of what might have been skipped
        if financial_entries and all_normalized_items:
            # Try to identify rows by common identifiers
            identifiers_to_check = ['Sr No.', 'Sr.No', 'Line', 'Doc No', 'Invoice Number', 'Document']
            
            # Check if entries have any of these identifiers
            for identifier in identifiers_to_check:
                if any(identifier in entry for entry in financial_entries):
                    # Build sets of the identifiers from original and normalized data
                    orig_identifiers = set()
                    for entry in financial_entries:
                        if identifier in entry and entry[identifier]:
                            orig_identifiers.add(str(entry[identifier]).strip())
                    
                    norm_identifiers = set()
                    for entry in all_normalized_items:
                        # Check both invoice and other document fields
                        if 'Invoice number' in entry and entry['Invoice number']:
                            norm_identifiers.add(str(entry['Invoice number']).strip())
                        if 'Other document number' in entry and entry['Other document number']:
                            norm_identifiers.add(str(entry['Other document number']).strip())
                    
                    missing = orig_identifiers - norm_identifiers
                    if missing:
                        logging.warning(f"Missing row identifiers (by {identifier}): {', '.join(sorted(missing))}")
                    
                    break  # Only check one identifier type that exists
    elif final_count > total_original_count + 1:  # +1 for possible payment entry
        logging.info(f"Got {final_count} normalized entries (more than original {total_original_count}), ")
        logging.info("This may include a payment entry and/or split entries which is acceptable")
    else:
        logging.info(f"Row count validation successful: {final_count} normalized entries from {total_original_count} original entries")
    
    logging.info(f"Successfully normalized {len(all_normalized_items)} items in total")
    return all_normalized_items

def process_batch(batch_data, global_metadata, llm, is_first_batch=True, retry_mode=False):
    """Process a batch of table data rows for normalization.
    
    Args:
        batch_data: List of financial entries to normalize
        global_metadata: Dictionary of metadata about the document
        llm: LLM provider instance
        is_first_batch: Whether this is the first batch (for payment entry creation)
        retry_mode: If True, use enhanced prompts to emphasize preserving all rows
        
    Returns:
        list: Normalized data entries
    """
    # Log batch size and contents
    entry_count = len(batch_data)
    logging.info(f"Processing batch with {entry_count} financial entries")
    logging.debug(f"First entry in batch: {json.dumps(batch_data[0], indent=2) if batch_data else 'None'}")
    logging.debug(f"Global metadata keys: {list(global_metadata.keys())}")
    
    # Check for payment/UTR information in global metadata
    has_payment_info = False
    payment_details = {}
    
    # Payment info extraction
    payment_utr = None
    payment_amount = None
    has_payment_info = False
    payment_details = {'utr': None, 'amount': None, 'date': None}
    
    # Check for payment number
    if global_metadata.get('payment_utr'):
        payment_utr = global_metadata.get('payment_utr')
    elif global_metadata.get('payment_number'):
        payment_utr = global_metadata.get('payment_number')
    
    # Check for payment amount
    payment_amount = global_metadata.get('payment_amount')
    if payment_amount and isinstance(payment_amount, str):
        # Remove currency symbols, commas and other non-numeric characters
        cleaned_amount = ''
        for c in payment_amount:
            if c.isdigit() or c == '.':
                cleaned_amount += c
        try:
            payment_amount = float(cleaned_amount) if cleaned_amount else None
            logging.info(f"Cleaned payment amount: {payment_amount}")
        except ValueError:
            logging.warning(f"Could not convert payment amount '{payment_amount}' to float")
            payment_amount = None  # Don't use hardcoded fallbacks
    
    # Use payment information if either UTR or amount is available
    if payment_utr or payment_amount:
        has_payment_info = True
        payment_details['utr'] = payment_utr
        payment_details['amount'] = payment_amount
        logging.info(f"Found payment information in metadata: UTR={payment_utr}, Amount={payment_amount}")
    elif global_metadata.get('payment_number'):
        # Fallback to payment_number if it exists
        has_payment_info = True
        payment_details['utr'] = global_metadata.get('payment_number')
        payment_details['amount'] = None
        logging.info(f"Using payment number as UTR: {payment_details['utr']}")
    
    # If we found a payment advice number but no UTR, use that as a fallback
    if not payment_details['utr'] and global_metadata.get('payment_advice_number'):
        payment_details['utr'] = global_metadata.get('payment_advice_number')
        logging.info(f"Using payment advice number as UTR: {payment_details['utr']}")
        has_payment_info = True
                
    # Define payment instruction based on payment info
    if has_payment_info:
        if is_first_batch:
            # Use f-strings instead of string.format to avoid issues with nested quotes
            payment_instruction = f"""
Create a new 'BDPO' type entry with payment details (only for the first batch):
- Use UTR: {payment_details['utr']}
- Use Amount: {payment_details['amount']}
- Set "Entry type" to "BDPO"
- Place UTR value in "Other document number"
"""
        else:
            payment_instruction = """Do NOT create any new payment entries. Only normalize existing entries."""
    else:
        payment_instruction = """No payment information found. Only normalize existing entries."""
                
    # Add stronger emphasis if this is a retry
    emphasis = ""
    if retry_mode:
        emphasis = """
⚠️ CRITICAL WARNING: PREVIOUS ATTEMPT FAILED TO PRESERVE ALL ROWS
⚠️ YOU MUST PRESERVE EVERY SINGLE ROW FROM THE INPUT - NO EXCEPTIONS
⚠️ DO NOT COMBINE, MERGE OR SKIP ANY ROWS - ONE OUTPUT ROW FOR EACH INPUT ROW EXACTLY
"""
        logging.info("Running in retry mode with enhanced preservation emphasis")
    
    # First create a partial prompt string without fields requiring formatting
    normalize_prompt_base = f"""
You are a payment advice normalization engine. Your task is to process financial entries and output standardized JSON objects.
{emphasis}
CRITICAL REQUIREMENTS:
1. PROCESS ALL INPUT ENTRIES - DO NOT SKIP ANY ROWS FOR ANY REASON
2. Ensure each input entry has a corresponding output entry in the EXACT SAME order
3. Format all entries according to the 9-field schema below
4. Verify row count matches exactly - you MUST process every entry
5. Handle 'bdpo' entries correctly - they must be 'BDPO' type with document in 'Other document number'
6. {payment_instruction}

INPUT:"""
    
    # Add the fields with nested braces separately without using f-string formatting
    normalize_prompt = normalize_prompt_base + """
- Financial entries: {batch_data}
- Metadata: {global_metadata}

ENTRY PROCESSING RULES:
1. Entry Types:
   - "BDPO": For ANY entry containing 'bdpo', 'co-op', 'coop', 'BD-Coop', or 'marketing' (CRITICAL: Check both invoice number AND description fields)
   - "TDS": For tax deduction entries (typically with -TDS- in the document number)
   - "Debit note": For debits, returns, damage
   - "Bank receipt": For payment entries (look in BOTH metadata AND table entries for payment info)
   - "Invoice": For all remaining invoice entries

2. Document Fields:
   - If an entry has a "Sr No." or similar field, preserve this value in your output
   - Keep track of each entry by its position/index to ensure none are skipped
   - Document numbers should go in appropriate field based on type (Invoice vs Other)
   
OUTPUT SCHEMA (each entry MUST have these fields):
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

YOUR RESPONSE MUST CONTAIN EXACTLY {entry_count} OBJECTS PLUS OPTIONAL PAYMENT ENTRY.
Do not skip any entries, even if they appear incomplete or unclear. Make best-effort normalization of all rows.

Response Format: JSON array containing normalized objects (plus optional payment entry)

FINAL CHECK - VERIFY ALL OF THESE BEFORE SUBMITTING:
- Array length should match the input entries (or +1 if payment entry was added)
- YOU MUST INCLUDE ALL ORIGINAL ROWS FROM INPUT
- Each object must have all 9 fields (null for missing values)
- JSON must be valid
- Any row with a Sr.No, Line No, or similar identifier should preserve that in the Invoice number or Other document number field"""
    
    # Payment instruction already defined earlier - we'll just update it with more specifics
    # if needed based on batch information
    
    # Only add payment entry for the first batch
    if not is_first_batch:
        # Override payment instruction to ensure no duplicate entries
        payment_instruction = "Do NOT create any new payment entries. Only normalize existing entries."
    
    # Fill in prompt template using string substitution to avoid format string conflicts
    # First convert our data to JSON strings
    batch_data_json = json.dumps(batch_data, indent=2)
    global_metadata_json = json.dumps(global_metadata, indent=2)
    
    # Now perform the string replacements
    filled_prompt = normalize_prompt.replace("{batch_data}", batch_data_json)
    filled_prompt = filled_prompt.replace("{global_metadata}", global_metadata_json)
    filled_prompt = filled_prompt.replace("{payment_instruction}", payment_instruction)
    
    messages = [HumanMessage(content=filled_prompt)]
    
    start_time = time.time()
    try:
        logging.info(f"Sending batch to LLM (entries: {entry_count}, prompt length: {len(filled_prompt)})")
        messages = [
            {"role": "system", "content": "You are an expert financial data normalization AI assistant."},
            {"role": "user", "content": filled_prompt}
        ]
        response = llm.invoke(messages)
        
        # Get LLM response content
        llm_response = response.content if hasattr(response, 'content') else str(response)
        
        # Log timing and response preview
        logging.info(f"LLM processing time: {time.time() - start_time:.2f} seconds")
        logging.debug(f"LLM response preview: \n{llm_response[:500]}...")
        
        # Save the raw LLM response for debugging
        debug_file = f"batch_response_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(debug_file, 'w') as f:
            f.write(llm_response)
        logging.info(f"Saved raw batch response to {debug_file}")
        
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
            
            # Verify entry count - more flexible checking to handle additional entries
            if len(normalized_batch) < entry_count:
                logging.warning(f"Expected at least {entry_count} entries but received only {len(normalized_batch)} entries from LLM")
            elif len(normalized_batch) > entry_count + 1:
                # If we got more than expected+1, it might be because the LLM split some entries
                # This is actually fine - just log it
                logging.info(f"Received {len(normalized_batch)} entries from LLM (expected around {entry_count})")
            
            # Check if the batch contains a payment entry
            has_payment_entry = False
            for entry in normalized_batch:
                if isinstance(entry, dict) and entry.get('Entry type') == 'Bank receipt':
                    logging.info("Found payment entry in the batch")
                    has_payment_entry = True
                    break
            
            # More relaxed validation - if we have fewer entries than expected but no validation error occurred,
            # we proceed anyway since the LLM might have combined or filtered some entries
            if len(normalized_batch) < entry_count - 1:  # Allow for some flexibility
                logging.warning(f"Entry count issue: Expected at least {entry_count-1} entries, got only {len(normalized_batch)}")
            else:
                logging.info(f"Processing batch with {len(normalized_batch)} entries")
            
            # Define allowed counts for compatibility with existing code
            allowed_counts = [entry_count, entry_count + 1]  # Allow exact count or +1 for payment entry
            # Log warning only if we still don't have the right count after unwrapping
            if len(normalized_batch) not in allowed_counts:
                logging.warning(f"After unwrapping, still have count mismatch. Got: {len(normalized_batch)}")
            
            debug_file = f"debug_llm_response_{time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(debug_file, 'w') as f:
                json.dump({
                    "prompt": filled_prompt,
                    "response": response.content,
                    "parsed": parsed_llm_output
                }, f, indent=2)
            logging.warning(f"Saved debug info to {debug_file}")
            
            return normalized_batch
            
        except json.JSONDecodeError as je:
            logging.error(f"Failed to parse LLM response: {str(je)}")
            logging.error(f"Raw response: {response.content}")
            return []
            
    except Exception as e:
        logging.error(f"Batch processing error: {str(e)}")
        logging.warning(f"Continuing with next batch despite error")
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
