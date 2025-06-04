#!/usr/bin/env python3
"""
LLM Normalization Engine for Payment Advice Extraction System.
Converts raw extracted data to structured format using OpenAI.
"""
import json
import os
import time
import datetime
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
    batch_size = 10  # Adjust as needed
    all_normalized_items = []
    
    if not financial_entries:
        logging.warning("No financial entries to normalize.")
        # Depending on requirements, you might return an empty list or global_metadata wrapped in a list
        # For consistency with previous behavior when table_data was empty, let's consider what's appropriate.
        # If the goal is a list of *normalized items*, and there are none, an empty list is suitable.
        return [] 
    
    table_data = financial_entries # Use financial_entries for iteration
    for i in range(0, len(table_data), batch_size):
        batch_data = table_data[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(table_data) + batch_size - 1) // batch_size
        
        logging.info(f"Processing batch {batch_num}: items {i+1}-{min(i+batch_size, len(table_data))} of {len(table_data)}")
        
        # Process this batch
        batch_results = process_batch(batch_data, global_metadata, llm)
        logging.debug(f"Batch {batch_num} results from process_batch (first 2 items): {batch_results[:2]}")
        logging.debug(f"Number of items in batch_results: {len(batch_results) if batch_results else 0}")

        # If batch processing failed to include invoice-specific details, manually combine
        # Check if batch_results are sufficiently detailed (e.g., contain invoice numbers or amounts)
        missing_details = True
        if batch_results and all(isinstance(item, dict) for item in batch_results):
            for item_idx, item in enumerate(batch_results):
                # Check for presence of key fields based on our updated field structure
                has_document_number = 'Document number' in item and item['Document number'] is not None and str(item['Document number']).strip() != ""
                has_amount_settled = 'Amount settled' in item and item['Amount settled'] is not None
                has_entry_type = 'Entry type' in item and item['Entry type'] is not None and str(item['Entry type']).strip() != ""
                logging.debug(f"  Item {item_idx} in batch: Document number present: {has_document_number}, Amount settled present: {has_amount_settled}, Entry type present: {has_entry_type}")
                logging.debug(f"  Item {item_idx} content: {item}")
                if has_document_number or has_amount_settled or has_entry_type:
                    missing_details = False
                    logging.debug(f"Batch {batch_num} deemed detailed based on item {item_idx}.")
                    break # Found at least one detailed item, so the batch is considered detailed
            if missing_details and batch_results: # If loop finished and no detailed item was found
                logging.debug(f"Batch {batch_num} still considered missing details after checking all {len(batch_results)} items.")
        else:
            logging.debug(f"Batch {batch_num} is empty or not a list of dicts, considered missing details.")

        logging.debug(f"Batch {batch_num} - missing_details flag: {missing_details}")
        
        if missing_details:
            logging.warning("Batch processing returned only global metadata. Manually combining with row data.")
            enhanced_batch = []
            # Process all rows in this batch
            for j, row in enumerate(batch_data):
                # If the batch_results has enough items, use them; otherwise duplicate the first one
                if j < len(batch_results):
                    item = batch_results[j].copy()  # Start with global metadata
                else:
                    # If we have fewer results than rows, duplicate the first result for remaining rows
                    item = batch_results[0].copy() if batch_results else global_metadata.copy()
                # Add row-specific data using the new field structure
                doc_number = row.get('Invoice Number', None) or row.get('Document Number', None)
                item['Document number'] = doc_number

                # Format date correctly
                date_str = row.get('Date', None) or row.get('Invoice Date', None)
                if date_str:
                    try:
                        # Try to parse common date formats
                        for date_format in ["%d-%b-%Y", "%d-%B-%Y", "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d"]:
                            try:
                                parsed_date = datetime.datetime.strptime(date_str, date_format)
                                item['Invoice date'] = parsed_date.strftime("%Y-%m-%d")  # Use ISO format
                                break
                            except ValueError:
                                continue
                        # If all formats failed, just use the original string
                        if 'Invoice date' not in item or not item['Invoice date']:
                            item['Invoice date'] = date_str
                    except Exception:
                        item['Invoice date'] = date_str
                else:
                    item['Invoice date'] = None

                # Handle amount settled
                amount_str = row.get('Amount Paid', '0') or row.get('Amount', '0')
                if amount_str:
                    # Remove parentheses, commas, and currency symbols
                    amount_str = amount_str.replace(',', '')
                    amount_str = amount_str.replace('₹', '').replace('$', '').replace('€', '').replace('£', '')
                    # Handle negative amounts in parentheses
                    if amount_str.startswith('(') and amount_str.endswith(')'):
                        amount_str = '-' + amount_str[1:-1]
                    try:
                        amount_value = float(amount_str)
                        item['Amount settled'] = amount_value
                    except ValueError:
                        item['Amount settled'] = None
                
                # Determine entry type and transaction type based on available data
                # Collect all available description fields for better inference
                description = ''
                for key, value in row.items():
                    if isinstance(value, str) and ('descr' in key.lower() or 'detail' in key.lower() or 'narration' in key.lower()):
                        description += str(value) + ' '
                
                # Add any other fields that might contain useful information
                description += ' ' + str(row.get('Description', ''))
                description += ' ' + str(row.get('Invoice description', ''))
                description += ' ' + str(row.get('Particulars', ''))
                description += ' ' + str(row.get('Remarks', ''))
                description = description.strip()
                
                # Initialize entry_type and document numbers
                entry_type = None
                transaction_type = None
                invoice_number = None
                other_doc_number = None
                
                # Extract potential document numbers based on patterns
                potential_inv_numbers = []
                potential_doc_numbers = []
                
                # Look for invoice number patterns (INV followed by digits, or digits with specific prefixes)
                for key, value in row.items():
                    if not value or not isinstance(value, str):
                        continue
                    
                    # Look for invoice patterns
                    if ('invoice' in key.lower() or 'inv' in key.lower()) and any(c.isdigit() for c in value):
                        potential_inv_numbers.append(value)
                    
                    # Look for UTR, BDPO, or other document number patterns
                    if ('utr' in key.lower() or 'ref' in key.lower() or 'tds' in key.lower() or 
                        'bdpo' in key.lower() or 'number' in key.lower() or 'certificate' in key.lower()):
                        potential_doc_numbers.append(value)
                
                # The best document number is the one with the most digits
                best_inv_number = None
                if potential_inv_numbers:
                    best_inv_number = max(potential_inv_numbers, key=lambda x: sum(c.isdigit() for c in x))
                
                best_doc_number = None
                if potential_doc_numbers:
                    best_doc_number = max(potential_doc_numbers, key=lambda x: sum(c.isdigit() for c in x))
                
                # Determine entry type from description and available information
                if 'TDS' in description or any('tds' in str(k).lower() for k in row.keys()):
                    entry_type = 'TDS dr'
                    transaction_type = 'Dr'
                    other_doc_number = best_doc_number or doc_number or 'TDS-' + str(row.get('Date', ''))
                
                # Check if this is an invoice
                elif ('INV' in description or 'Invoice' in description or 
                      any('invoice' in str(k).lower() for k in row.keys()) or 
                      best_inv_number):
                    entry_type = 'Invoice cr'
                    transaction_type = 'Cr'
                    invoice_number = best_inv_number or doc_number
                
                # Check if this is a bank receipt/payment
                elif ('UTR' in description or 'Payment' in description or 'Receipt' in description or 
                      'bank' in description.lower() or 'paid' in description.lower() or 
                      any('utr' in str(k).lower() or 'payment' in str(k).lower() for k in row.keys())):
                    entry_type = 'Bank receipt dr'
                    transaction_type = 'Dr'
                    other_doc_number = best_doc_number or doc_number or 'UTR-' + str(row.get('Date', ''))
                
                # Check if this is a BDPO entry
                elif 'BDPO' in description or 'BD-Coop' in description:
                    entry_type = 'BDPO dr'
                    transaction_type = 'Dr'
                    other_doc_number = best_doc_number or doc_number or 'BDPO-' + str(row.get('Date', ''))
                
                # Check if this is a debit note
                elif 'Debit' in description or 'DN' in description:
                    entry_type = 'Debit note dr'
                    transaction_type = 'Dr'
                    other_doc_number = best_doc_number or doc_number or 'DN-' + str(row.get('Date', ''))
                
                else:
                    # Default fallback based on amount direction
                    amount_str = row.get('Amount Paid', '0') or row.get('Amount', '0')
                    amount_settled = None
                    
                    if isinstance(amount_str, str):
                        # Clean the amount string
                        amount_str = amount_str.replace(',', '')
                        amount_str = amount_str.replace('₹', '').replace('$', '').replace('€', '').replace('£', '')
                        # Handle negative amounts in parentheses
                        if amount_str.startswith('(') and amount_str.endswith(')'):
                            amount_str = '-' + amount_str[1:-1]
                        try:
                            amount_settled = float(amount_str)
                        except ValueError:
                            pass
                    elif isinstance(amount_str, (int, float)):
                        amount_settled = float(amount_str)
                    
                    if amount_settled is not None and amount_settled < 0:
                        entry_type = 'Invoice cr'
                        transaction_type = 'Cr'
                        invoice_number = best_inv_number or 'INV-' + str(row.get('Date', ''))
                    else:
                        entry_type = 'Bank receipt dr'
                        transaction_type = 'Dr'
                        other_doc_number = best_doc_number or 'UTR-' + str(row.get('Date', ''))
                
                # Set the entry type, transaction type and document numbers
                item['Entry type'] = entry_type
                item['Transaction type(Dr/cr)'] = transaction_type
                
                # Set document numbers based on entry type
                if entry_type == 'Invoice cr':
                    item['Invoice number'] = invoice_number
                    item['Other document number'] = None
                else:
                    item['Invoice number'] = None
                    item['Other document number'] = other_doc_number
                
                # Set vendor and customer names (correctly mapped)
                item['Customer Name as per Payment advice'] = item.get('Customer Name as per Payment advice') or row.get('Customer Name', None) or global_metadata.get('customer_name', None)
                item['Vendor name (Payee name)'] = item.get('Vendor name (Payee name)') or row.get('Vendor Name', None) or global_metadata.get('vendor_name', None)
                
                enhanced_batch.append(item)
            batch_results = enhanced_batch
            
        all_normalized_items.extend(batch_results)
    
    logging.info(f"Successfully normalized {len(all_normalized_items)} items in total")
    
    # The new normalize_prompt asks for specific Title Cased field names directly.
    # The fallback logic also tries to create fields like 'invoice_number', 'invoice_date', 'amount_paid'.
    # The process_batch function is expected to return a list of dictionaries where keys are already the target schema names.

    # Define the expected fields from the LLM response (9 fields as per prompt)
    final_expected_fields = [
        "Payment Advice number", "Sendor mail", "Original sendor mail", 
        "Vendor name (Payee name)", "Customer Name as per Payment advice", 
        "Entry type", "Amount settled", "Other document number", "Invoice number"
    ]

    # Mapping for keys potentially coming from fallback or if LLM deviates from strict output format
    fallback_to_final_mapping = {
        'invoice_number': 'Invoice number',
        'amount_paid': 'Amount settled',
        'doc_type': 'Entry type',
        'doc_number': None,  # This needs special handling based on entry type
        'utr_number': 'Other document number',  # For bank receipt entries
        'tds_reference': 'Other document number',  # For TDS entries
        'bdpo_number': 'Other document number',  # For BDPO entries
        'debit_note_number': 'Other document number',  # For Debit note entries
        'payment_advice_number': 'Payment Advice number',
        'sendor_mail': 'Sendor mail',
        'original_sendor_mail': 'Original sendor mail',
        'vendor_name': 'Vendor name (Payee name)',
        'customer_name': 'Customer Name as per Payment advice',
        'utr': 'UTR'
    }

    final_structured_items = []
    for item_llm in all_normalized_items: # item_llm is a dict from process_batch (LLM output)
        final_item = {}

        # Map LLM fields to schema_processor.FINAL_COLUMNS keys
        final_item["Payment date"] = None  # Filled later by create_final_df from global metadata
        final_item["Payment advice number"] = item_llm.get("Payment Advice number")
        final_item["Customer name"] = item_llm.get("Customer Name as per Payment advice")
        final_item["Customer id (SAP id)"] = None  # Not in line-item LLM output
        
        # Get entry type and set Doc type
        entry_type = item_llm.get("Entry type")
        final_item["Doc type"] = entry_type
        
        # Determine transaction type from entry type suffix for internal use
        transaction_type = None
        if entry_type:
            if entry_type.lower().endswith(" dr"):
                transaction_type = "Dr"
            elif entry_type.lower().endswith(" cr"):
                transaction_type = "Cr"
        
        # Handle document numbers based on entry type
        invoice_number = item_llm.get("Invoice number")
        other_doc_number = item_llm.get("Other document number")
        
        # Properly separate Invoice number and Other document number based on entry type
        if entry_type and entry_type.lower() == "invoice cr":
            # For Invoice entries:
            # - Set Invoice number field
            # - Keep Other document number as null
            # - Doc number gets the invoice number for consistency
            final_item["Invoice number"] = invoice_number
            final_item["Doc number"] = invoice_number  # For consistency in Doc number field
        else:
            # For non-Invoice entries:
            # - Keep Invoice number as null
            # - Set Other document number based on entry type
            # - Doc number gets the other document number for consistency
            final_item["Invoice number"] = None
            final_item["Doc number"] = other_doc_number
            
            # Set UTR specifically for bank receipts
            if entry_type and entry_type.lower() == "bank receipt dr":
                final_item["UTR"] = other_doc_number
            else:
                final_item["UTR"] = None
        
        # Fallback logic if document numbers are missing
        if not final_item["Doc number"] and entry_type:
            # Try to extract from other fields based on entry type
            if entry_type.lower() == "invoice cr":
                doc_num = item_llm.get("invoice_number")
                final_item["Doc number"] = doc_num
                final_item["Invoice number"] = doc_num
            elif entry_type.lower() == "bank receipt dr":
                doc_num = item_llm.get("utr") or item_llm.get("payment_reference") or item_llm.get("UTR no.")
                final_item["Doc number"] = doc_num
                final_item["UTR"] = doc_num
            elif entry_type.lower() == "bdpo dr":
                doc_num = item_llm.get("bdpo_number") or item_llm.get("BDPO inv no.")
                final_item["Doc number"] = doc_num
            elif entry_type.lower() == "debit note dr":
                doc_num = item_llm.get("debit_note_number") or item_llm.get("Debit note no.")
                final_item["Doc number"] = doc_num
            elif entry_type.lower() == "tds dr":
                doc_num = item_llm.get("tds_reference") or item_llm.get("tds_certificate_number") or item_llm.get("TDS ref")
                final_item["Doc number"] = doc_num

        # Handle TDS entries
        if entry_type and entry_type.lower() == "tds dr":
            tds_amount = item_llm.get("Amount settled")
            if tds_amount is not None:
                try:
                    tds_amount = float(tds_amount)
                except (ValueError, TypeError):
                    tds_amount = 0.0
                # TDS amount should be positive in the output since it's a dr entry
                final_item["TDS deducted"] = tds_amount if tds_amount > 0 else abs(tds_amount)
                final_item["Amount paid"] = None
            else:
                final_item["TDS deducted"] = 0.0
                final_item["Amount paid"] = None
        else:
            # Handle normal payment entries
            amount = item_llm.get("Amount settled")
            if amount is not None:
                try:
                    amount = float(amount)
                except (ValueError, TypeError):
                    amount = 0.0
                # Keep amount as is without flipping signs
                final_item["Amount paid"] = amount
            else:
                final_item["Amount paid"] = 0.0
            final_item["TDS deducted"] = 0.0

        # Set invoice date if available in the item_llm, otherwise it stays null from above
        if "Invoice date" in item_llm and item_llm["Invoice date"]:
            final_item["Invoice date"] = item_llm.get("Invoice date")
        else:
            final_item["Invoice date"] = None
            
        # Set currency and PA link defaults
        final_item["Currency"] = "INR"     # Default currency
        final_item["PA link"] = None       # Default PA link
        amount_settled_str = item_llm.get("Amount settled")
        amount_settled_numeric = None
        if amount_settled_str is not None:
            try:
                amount_settled_numeric = float(amount_settled_str)
            except (ValueError, TypeError):
                logging.warning(f"Could not convert 'Amount settled' value '{amount_settled_str}' to float. Item: {item_llm}")
        
        # Determine if this is a TDS entry
        is_tds_entry = entry_type and "tds" in entry_type.lower()
        
        if is_tds_entry:
            # For TDS entries, amount goes to TDS deducted
            final_item["TDS deducted"] = abs(amount_settled_numeric) if amount_settled_numeric is not None else None
            final_item["Amount paid"] = 0.0
        else:
            # For all other entries, amount goes to Amount paid
            final_item["Amount paid"] = abs(amount_settled_numeric) if amount_settled_numeric is not None else None
            final_item["TDS deducted"] = 0.0
        
        # Add currency and PA link fields
        final_item["Currency"] = "INR"  # Default to INR - can be overridden in create_final_df if needed
        final_item["PA link"] = None  # Not in line-item LLM output

        final_structured_items.append(final_item)
        
    logging.info(f"Normalization complete. Total items structured: {len(final_structured_items)}")
    logging.debug(f"First 2 items from final_structured_items before return: {final_structured_items[:2]}")
    return final_structured_items

def process_batch(batch_data, global_metadata, llm):
    """
    Process a batch of table data rows for normalization.

    Args:
        batch_data: A subset of the table data rows.
        global_metadata: The extracted global metadata.
        llm: The language model instance.
        
    Returns:
        A list of normalized invoice items for this batch.
    """
    # Create the normalization prompt for this batch
    normalize_prompt = """
You are a payment advice normalization engine. Your task is to process a list of financial entries.
For EACH financial entry provided in the `FINANCIAL ENTRIES TO PROCESS` list below, you MUST convert it into the STANDARD FORMAT.
After processing ALL entries, you MUST gather them into a single JSON array. Each entry from the input list must have a corresponding JSON object in the output array.

BUSINESS CONTEXT:
The system is processing payment advice documents received from clients. You (the vendor) receive these payment records from your clients (buyers). These documents show entries in the client's accounting system that affect your account with them. This data needs to be normalized into a consistent structure for your accounting system.

GLOBAL METADATA:
{global_metadata}

FINANCIAL ENTRIES TO PROCESS:
{batch_data}

REQUIRED OUTPUT FIELDS:
1. Payment Advice number - Reference number of the document being processed
2. Sendor mail - Email address from which the payment advice was received
3. Original sendor mail - Root sender email (ignoring any forwards)
4. Vendor name (Payee name) - Your company name as the vendor/payee
5. Customer Name as per Payment advice - Client name as shown in the payment advice
6. Entry type - Must be one of: "Bank receipt dr", "BDPO dr", "Debit note dr", "TDS dr", "Invoice cr"
7. Amount settled - Monetary value associated with the entry
8. Other document number - The unique identifier for non-invoice entries
9. Invoice number - Only populated for Invoice entries, otherwise null

PROCESSING RULES:
1. For EACH entry:
   - Determine "Entry type" by analyzing the transaction details, descriptions, and type of numbers mentioned:
      • "Bank receipt dr" = Bank payment references/receipts, UTR numbers, often containing "paid", "payment", "receipt", "bank"
      • "BDPO dr" = Marketing service credits, often containing "BDPO", "BD-Coop", "marketing"
      • "Debit note dr" = Debit notes, often containing "debit", "damage", "return", "DN"
      • "TDS dr" = Tax Deducted at Source, often containing "TDS", "tax", "deduction", "certificate"
      • "Invoice cr" = Invoice entries, often containing "INV", "invoice", "bill", "sale"

2. For document number fields:
   • "Invoice number": ONLY populate this field when the Entry type is "Invoice cr". Look for numbers prefixed with INV, invoice numbers, bill numbers. Must be non-null for Invoice entries.
   • "Other document number": For non-Invoice entries, populate with:
      - For "Bank receipt dr": Use UTR or payment reference number (often numeric and 10+ digits)
      - For "BDPO dr": Use BDPO invoice number (often prefixed with BDPO)
      - For "Debit note dr": Use debit note number (often prefixed with DN)
      - For "TDS dr": Use TDS reference or certificate number
      - For "Invoice cr": Leave as null

3. For "Amount settled":
   • Convert the source amount to a numerical value (remove currency symbols like ₹, $, and commas)
   • For values in parentheses like (1,234.56), treat as negative values
   • DO NOT flip the signs of amounts - preserve them exactly as they appear in the source data
   • "dr" entries (debits) are typically positive amounts in client books (they receive money)
   • "cr" entries (credits) are typically negative amounts in client books (they give money)
   • NEVER leave this field as null - provide a numerical value even if it's an estimate
   • For TDS entries, look for small percentage amounts (usually 1-10% of invoice values)

DATA PROCESSING STRATEGY:
1. First, scan for obvious entry types based on keywords in descriptions or reference numbers
2. Group related invoice and payment entries by looking at matching amounts, dates, or references
3. Ensure every entry has an appropriate document number - use Invoice number for Invoice cr entries, and Other document number for all other entry types
4. Use context from surrounding entries to infer missing values (e.g., Payment Advice numbers should be consistent across related entries)
5. If an amount appears both as a positive and negative value in different entries, they likely represent offsetting debit/credit pairs

CLARIFICATION OF TRANSACTION TYPES:
From the client's perspective (buyers):
- Debit entries (dr/+ve) represent money they receive or credits in their favor
- Credit entries (cr/-ve) represent money they give to the vendor
- TDS entries are DEBITS because the client pays tax on behalf of the vendor
- Invoice entries are CREDITS because the client owes money to the vendor
- Bank receipts are DEBITS because they offset the client's credit to the vendor

OUTPUT FORMAT (STRICT JSON):
[
  { 
    "Payment Advice number": "337027030", 
    "Sendor mail": "accounts@clientcompany.com", 
    "Original sendor mail": "finance@clientcompany.com",
    "Vendor name (Payee name)": "Vendor Company Ltd", 
    "Customer Name as per Payment advice": "KWICK LIVING (I) PRIVATE LIMITED",
    "Entry type": "Invoice cr",
    "Amount settled": -5000.00, 
    "Other document number": null, 
    "Invoice number": "INV-001"
  },
  { 
    "Payment Advice number": "337027030", 
    "Sendor mail": "accounts@clientcompany.com", 
    "Original sendor mail": "finance@clientcompany.com",
    "Vendor name (Payee name)": "Vendor Company Ltd", 
    "Customer Name as per Payment advice": "KWICK LIVING (I) PRIVATE LIMITED",
    "Entry type": "Bank receipt dr",
    "Amount settled": 5000.00, 
    "Other document number": "UTR763950212", 
    "Invoice number": null
  },
  { 
    "Payment Advice number": "337027030", 
    "Sendor mail": "accounts@clientcompany.com", 
    "Original sendor mail": "finance@clientcompany.com",
    "Vendor name (Payee name)": "Vendor Company Ltd", 
    "Customer Name as per Payment advice": "KWICK LIVING (I) PRIVATE LIMITED",
    "Entry type": "TDS dr",
    "Amount settled": 500.00, 
    "Other document number": "TDS-CERT-123", 
    "Invoice number": null
  }
]

CRITICAL (JSON Array Output):
- The final output MUST be a JSON array.
- Every extraction MUST have ALL 9 fields specified.
- Never skip any entry from the input list; all entries must be represented in the output.
- NEVER produce invalid JSON with trailing commas, unbalanced brackets, or incorrect syntax.
- Look carefully at the data for document numbers and record each one - NEVER leave document number fields empty unless absolutely nothing in the source could be a document number.
- For every "Invoice cr" entry, ALWAYS populate the Invoice number field and leave Other document number as null.
- For all non-Invoice entries, ALWAYS populate the Other document number field and leave Invoice number as null.
"""
    
    filled_prompt = normalize_prompt.replace("{global_metadata}", json.dumps(global_metadata, indent=2))
    filled_prompt = filled_prompt.replace("{batch_data}", json.dumps(batch_data, indent=2))
    
    messages = [HumanMessage(content=filled_prompt)]
    
    start_time = time.time()
    try:
        logging.info(f"Sending batch normalization prompt to LLM (length: {len(filled_prompt)} chars)")
        response = llm.invoke(messages)
        api_time = time.time() - start_time
        logging.info(f"Batch processed in {api_time:.2f} seconds")
        logging.debug(f"Raw LLM response for batch normalization: {response.content}") # Log raw response
        
        # Parse the response
        parsed_llm_output = json.loads(response.content)
        
        # Attempt to extract the list of items, accommodating various wrapper structures
        if isinstance(parsed_llm_output, list):
            normalized_batch = parsed_llm_output
        elif isinstance(parsed_llm_output, dict):
            # Check for common wrapper keys that might contain the list of items
            potential_wrapper_keys = ['result', 'results', 'items', 'data', 'entries', 'normalized_items', 'normalized_entries']
            extracted_list = False
            for key in potential_wrapper_keys:
                if key in parsed_llm_output and isinstance(parsed_llm_output[key], list):
                    normalized_batch = parsed_llm_output[key]
                    extracted_list = True
                    break
            
            if not extracted_list:
                # If it's a dictionary but doesn't contain a recognized list under a common key,
                # assume the dictionary itself is a single item (e.g., LLM returned one item not in an array).
                normalized_batch = [parsed_llm_output]
        else:
            # If the output is neither a list nor a dict (e.g., a string from a misconfigured LLM or unexpected format)
            logging.warning(f"LLM returned an unexpected data type: {type(parsed_llm_output)}. Content: {response.content}. Wrapping as a single item list.")
            # Attempt to wrap it in a list; this might be an error case depending on strictness required.
            normalized_batch = [parsed_llm_output] 

        # Final safeguard: Ensure normalized_batch is a list. 
        # This primarily handles cases where the LLM might return a single item not in a list and not wrapped, 
        # and the above logic somehow resulted in a non-list (though it aims to always produce a list).
        if not isinstance(normalized_batch, list):
            logging.warning(f"Post-processing resulted in a non-list. Original type: {type(parsed_llm_output)}, Current type: {type(normalized_batch)}. Wrapping in a list.")
            normalized_batch = [normalized_batch]
        
        logging.info(f"Successfully normalized {len(normalized_batch)} items in this batch")
        return normalized_batch
        
    except Exception as e:
        logging.error(f"Error during batch normalization: {str(e)}")
        # Return empty list for this batch
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
