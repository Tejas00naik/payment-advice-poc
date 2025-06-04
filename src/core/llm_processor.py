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
                # Check for presence of key invoice-specific fields
                has_invoice_number = 'Invoice number' in item and item['Invoice number'] is not None and str(item['Invoice number']).strip() != ""
                has_amount_settled = 'Amount settled' in item and item['Amount settled'] is not None
                # logging.debug(f"  Item {item_idx} in batch: Invoice number present: {has_invoice_number}, Amount settled present: {has_amount_settled}")
                # logging.debug(f"  Item {item_idx} content: {item}")
                if has_invoice_number or has_amount_settled:
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
                # Add row-specific data
                item['invoice_number'] = row.get('Invoice Number', None)
                # Format date correctly as DD-MM-YYYY
                date_str = row.get('Date', None)
                if date_str:
                    try:
                        # Try to parse common date formats
                        for date_format in ["%d-%b-%Y", "%d-%B-%Y", "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d"]:
                            try:
                                parsed_date = datetime.datetime.strptime(date_str, date_format)
                                item['invoice_date'] = parsed_date.strftime("%d-%m-%Y")
                                break
                            except ValueError:
                                continue
                        # If all formats failed, just use the original string
                        if 'invoice_date' not in item or not item['invoice_date']:
                            item['invoice_date'] = date_str
                    except Exception:
                        item['invoice_date'] = date_str
                else:
                    item['invoice_date'] = None
                # Handle amounts in parentheses as negative
                amount_str = row.get('Amount Paid', '0')
                if amount_str:
                    # Remove parentheses, commas, and currency symbols
                    amount_str = amount_str.replace(',', '')
                    amount_str = amount_str.replace('₹', '').replace('$', '').replace('€', '').replace('£', '')
                    # Handle negative amounts in parentheses
                    if amount_str.startswith('(') and amount_str.endswith(')'):
                        amount_str = '-' + amount_str[1:-1]
                    try:
                        item['amount_paid'] = float(amount_str)
                    except ValueError:
                        item['amount_paid'] = None
                # Check if this is a TDS entry
                if 'TDS' in str(item['invoice_number']) or 'TDS' in str(row.get('Invoice description', '')):
                    item['doc_type'] = 'TDS'
                    if item['amount_paid'] and item['amount_paid'] < 0:
                        item['tds_deducted'] = abs(item['amount_paid'])
                        item['amount_paid'] = None
                else:
                    item['doc_type'] = 'Invoice'
                
                enhanced_batch.append(item)
            batch_results = enhanced_batch
            
        all_normalized_items.extend(batch_results)
    
    logging.info(f"Successfully normalized {len(all_normalized_items)} items in total")
    
    # The new normalize_prompt asks for specific Title Cased field names directly.
    # The fallback logic also tries to create fields like 'invoice_number', 'invoice_date', 'amount_paid'.
    # The process_batch function is expected to return a list of dictionaries where keys are already the target schema names.

    # Define the expected final 11 fields as per the normalize_prompt for consistency check / final mapping.
    final_expected_fields = [
        "Payment Advice number", "Sendor mail", "Original sendor mail", "Vendor name", 
        "Customer Name as per Payment advice", "Entry type", "Invoice number", "Invoice date",
        "UTR no.", "BDPO inv no.", "Credit note no.", "Amount settled"
    ]

    # Mapping for keys potentially coming from fallback or if LLM deviates from strict Title Case output.
    # This maps commonly produced snake_case keys (especially from fallback) to the final Title Case fields.
    fallback_to_final_mapping = {
        'invoice_number': 'Invoice number',
        # 'invoice_date': 'Invoice Date', # 'Invoice Date' is not one of the 11 target fields in normalize_prompt.
                                        # If it's needed, it should be added to final_expected_fields and prompt.
        'amount_paid': 'Amount settled',
        'tds_deducted': 'TDS Deducted', # Assuming TDS might appear and needs mapping, not in 11 fields.
        'doc_type': 'Entry type',
        'doc_number': 'Invoice number', # Or 'Credit note no.' or 'BDPO inv no.' depending on context not available here.
                                        # 'Invoice number' is a safe default if 'doc_number' appears from fallback.
        # Global metadata fields that might be snake_case if not handled by LLM's direct output
        'payment_advice_number': 'Payment Advice number',
        'sendor_mail': 'Sendor mail',
        'original_sendor_mail': 'Original sendor mail',
        'vendor_name': 'Vendor name',
        'customer_name': 'Customer Name as per Payment advice', # Maps to the specific customer name field
        'utr': 'UTR no.'
        # Note: 'payment_date', 'customer_id', 'currency' are not in the 11 target fields of normalize_prompt.
        # They would be part of the global_metadata if extracted by the first LLM call and can be added if needed.
    }

    final_structured_items = []
    for item_llm in all_normalized_items: # item_llm is a dict from process_batch (LLM output)
        final_item = {}

        # Map LLM fields to schema_processor.FINAL_COLUMNS keys
        final_item["Payment date"] = None # Filled later by create_final_df from global metadata
        final_item["Payment advice number"] = item_llm.get("Payment Advice number")
        final_item["Invoice number"] = item_llm.get("Invoice number")
        final_item["Invoice date"] = item_llm.get("Invoice date")
        final_item["Customer name"] = item_llm.get("Customer Name as per Payment advice")
        final_item["Customer id (SAP id)"] = None # Not in line-item LLM output
        final_item["UTR"] = item_llm.get("UTR no.")
        
        entry_type = item_llm.get("Entry type")
        final_item["Doc type"] = entry_type

        doc_number_val = None
        if entry_type == "Invoice":
            doc_number_val = item_llm.get("Invoice number")
        elif entry_type == "BD-Coop Settlement":
            doc_number_val = item_llm.get("BDPO inv no.")
        elif entry_type in ["Debit Note", "Credit Note"]:
            doc_number_val = item_llm.get("Credit note no.")
        elif entry_type == "TDS": # TDS might reference an invoice number
            doc_number_val = item_llm.get("Invoice number") # Or another reference if available
        # For 'Payment' or other types, doc_number might be sourced differently or be None
        final_item["Doc number"] = doc_number_val

        amount_settled_str = item_llm.get("Amount settled")
        amount_settled_numeric = None
        if amount_settled_str is not None:
            try:
                amount_settled_numeric = float(amount_settled_str)
            except (ValueError, TypeError):
                logging.warning(f"Could not convert 'Amount settled' value '{amount_settled_str}' to float. Item: {item_llm}")
        
        if entry_type == "TDS":
            # TDS deducted is typically positive, representing the amount withheld.
            # LLM is prompted to return positive TDS amounts in 'Amount settled' for TDS entries.
            final_item["TDS deducted"] = abs(amount_settled_numeric) if amount_settled_numeric is not None else None
            final_item["Amount paid"] = 0.0
        else:
            # Amount paid for invoices/other non-TDS entries should be positive.
            # LLM returns negative 'Amount settled' for invoices (debits).
            final_item["Amount paid"] = abs(amount_settled_numeric) if amount_settled_numeric is not None else None
            final_item["TDS deducted"] = 0.0

        final_item["Currency"] = None # Filled later by create_final_df
        final_item["PA link"] = None # Not in line-item LLM output

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
For EACH financial entry provided in the `FINANCIAL ENTRIES TO PROCESS` list below, you MUST convert it into the STANDARD 11-FIELD FORMAT.
After processing ALL entries, you MUST gather them into a single JSON array. Each entry from the input list must have a corresponding JSON object in the output array.

GLOBAL METADATA:
{global_metadata}

FINANCIAL ENTRIES TO PROCESS:
{batch_data}

REQUIRED OUTPUT FIELDS:
1. Payment Advice number
2. Sendor mail
3. Original sendor mail
4. Vendor name
5. Customer Name as per Payment advice
6. Entry type (MUST be: Invoice/Payment/BD-Coop Settlement/Debit Note/TDS)
7. Invoice number
8. Invoice date (format as YYYY-MM-DD if possible, otherwise as seen)
9. UTR no.
10. BDPO inv no.
11. Credit note no.
12. Amount settled

PROCESSING RULES:
1. For EACH entry:
   - Determine "Entry type" by analyzing description:
        • "Invoice" = Invoice references
        • "Payment" = UTR numbers + payment keywords
        • "BD-Coop Settlement" = BD/Coop invoice numbers
        • "Debit Note" = Debit note references
        • "TDS" = TDS/Tax keywords

2. Extract field-specific data:
   - "Amount settled": 
        • Convert the source amount to a numerical value (remove currency symbols like ₹, $, and commas like in 1,234.56).
        • Ensure that amounts in parentheses, e.g., (1,234.56), are treated as negative numbers (e.g., -1234.56) after this initial numerical conversion.
        • APPLY SIGN TRANSFORMATION FOR THE 'Amount settled' OUTPUT FIELD:
            - If the numerically converted source amount (from the steps above) is NEGATIVE (this represents a credit to your vendor account in the customer's books, meaning it's money owed to you or a reduction of what you owe them): The final 'Amount settled' value in the output should be POSITIVE.
            - If the numerically converted source amount (from the steps above) is POSITIVE (this represents a debit to your vendor account in the customer's books, meaning it's money you owe them or a payment they made): The final 'Amount settled' value in the output should be NEGATIVE.
        (In simpler terms: after converting the source amount to a clean number, flip its sign to get the 'Amount settled' value for the output.)
   - "Invoice number": Only for Invoice/Debit Note types
   - "UTR no.": Only for Payment entries
   - "BDPO inv no.": Only for BD-Coop Settlement
   - "Credit note no.": If available

3. SIGN CONVENTION CLARIFICATION:
   - Vendor account in customer books:
        • Credit entry (our money) = Negative in source → POSITIVE in "Amount settled"
        • Debit entry (their money) = Positive in source → NEGATIVE in "Amount settled"

4. Use EXACTLY these names in output JSON array:
["Payment Advice number", "Sendor mail", "Original sendor mail", "Vendor name", 
"Customer Name as per Payment advice", "Entry type", "Invoice number", "Invoice date", 
"UTR no.", "BDPO inv no.", "Credit note no.", "Amount settled"]

OUTPUT FORMAT (STRICT JSON):
[
  { "Payment Advice number": "PA-123", "Sendor mail": "sender@client.com", "Invoice date": "2023-01-15", ... }},
  { "Payment Advice number": "PA-123", "Sendor mail": "sender@client.com", "Invoice date": "2023-01-16", ... }}
]

CRITICAL (JSON Array Output):
- The final output MUST be a JSON array.
- Each object in the array MUST represent one processed financial entry from the input `FINANCIAL ENTRIES TO PROCESS` list.
- Each object in the array MUST contain exactly 12 fields as specified in REQUIRED OUTPUT FIELDS.
- Use null for unavailable fields.
- Never skip any entry from the input list; if an entry cannot be fully processed, still include it with nulls for missing fields but ensure all 12 keys are present.
- APPLY SIGN CONVERSION: As specified in PROCESSING RULES, flip the sign of the source amount to get the 'Amount settled' value.
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
