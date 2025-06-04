#!/usr/bin/env python3
"""
Schema Processor for Payment Advice Extraction System.
Handles data validation, schema enforcement, and final DataFrame creation.
"""
import logging
import pandas as pd
from typing import List, Dict, Any
import datetime

# Define final schema columns
FINAL_COLUMNS = [
    "Payment Advice number", "Sendor mail", "Original sendor mail", 
    "Vendor name (Payee name)", "Customer Name as per Payment advice", 
    "Entry type", "Amount settled", "Other document number", "Invoice number"
]


def create_final_df(normalized_data, metadata=None, extracted_structure=None):
    """
    Create a validated DataFrame from normalized output.
    
    Args:
        normalized_data (list): List of normalized data items
        metadata (dict): Optional metadata to fill into the DataFrame
        extracted_structure (dict): Original extraction structure for validation
        
    Returns:
        pd.DataFrame: Final validated DataFrame
    """
    # First, validate we have all rows from the original extraction
    if extracted_structure and isinstance(extracted_structure, dict):
        try:
            # Count financial entries in the original extraction
            orig_entries = extracted_structure.get('table_data', {}).get('financial_entries', [])
            orig_count = len(orig_entries)
            norm_count = len(normalized_data)
            
            # Check the row counts
            if orig_count > 0 and norm_count < orig_count:
                logging.warning(f"Warning: Original extraction had {orig_count} rows but normalized data has only {norm_count} rows")
                logging.warning("Some rows may have been lost during normalization")
                
                # Attempt to check the row identifiers if they exist to see which rows were dropped
                if orig_count > 0 and len(orig_entries[0]) > 0:
                    # Check if entries have serial numbers
                    serial_fields = [f for f in orig_entries[0].keys() if f.lower() in ('sr no.', 'sr.no', 'sno', 's.no', '#', 'row', 'line')]
                    
                    if serial_fields:
                        serial_field = serial_fields[0]
                        orig_serials = set([str(entry.get(serial_field)) for entry in orig_entries if entry.get(serial_field)])
                        
                        # Build a corresponding set from normalized data if possible
                        norm_serials = set()
                        for entry in normalized_data:
                            # Check for serial number in different forms
                            for field in entry.keys():
                                if field.lower() in ('sr no.', 'sr.no', 'sno', 's.no', '#', 'row', 'line', 'invoice number'):
                                    if entry.get(field):
                                        norm_serials.add(str(entry.get(field)))
                                        break
                        
                        # Find missing serials
                        missing = orig_serials - norm_serials
                        if missing:
                            logging.warning(f"Missing row numbers: {', '.join(sorted(missing))}")
        except Exception as e:
            logging.error(f"Error validating row counts: {e}")
    
    # Create DataFrame from the normalized data
    df = pd.DataFrame(normalized_data)
    
    # Fill metadata columns if available
    if metadata and isinstance(metadata, dict):
        for key, value in metadata.items():
            # Skip None values to avoid error in fillna
            if value is not None:
                if key in df.columns:
                    df[key] = df[key].fillna(value)
                elif key not in df.columns:
                    df[key] = value
    
    # Log final column info
    logging.info(f"Created final DataFrame with {len(df)} rows and columns: {list(df.columns)}")
    
    # Ensure all columns exist
    for col in FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    
    # Post-processing: fill in metadata if values are not None
    if metadata.get("payment_advice_no") is not None:
        df["Payment Advice number"] = df["Payment Advice number"].fillna(metadata.get("payment_advice_no"))
    
    if metadata.get("customer_name") is not None:
        df["Customer Name as per Payment advice"] = df["Customer Name as per Payment advice"].fillna(metadata.get("customer_name"))
    
    # Type conversion for numeric fields
    for col in ["Amount settled"]:
        if col in df.columns:
            # Handle different formats (commas, currency symbols)
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace('[^0-9.-]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Ensure we only have the columns we want, in the right order
    return df[FINAL_COLUMNS]


def validate_df(df: pd.DataFrame) -> bool:
    """
    Ensure data quality before saving
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    # Skip validation if empty DataFrame
    if df.empty:
        return True
    
    checks = [
        # Check invoice numbers exist
        (df["Invoice number"].notnull().all(), "Missing invoice numbers"),
        
        # Check for consistency in payment advice number
        (df["Payment Advice number"].nunique() <= 1 or df["Payment Advice number"].isnull().all(), 
         "Multiple payment advice numbers"),
        
        # Check for negative total amount (could be valid in some cases)
        (not (df["Amount settled"].sum() < 0), "Negative total amount"),
        
        # Check for null values in key fields
        ((df["Amount settled"].notnull()).any(), "No valid amounts found")
    ]
    
    for valid, msg in checks:
        if not valid:
            raise ValueError(f"Validation failed: {msg}")
    
    return True


def deduplicate_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate entries based on key fields
    
    Args:
        df: DataFrame to deduplicate
        
    Returns:
        Deduplicated DataFrame
    """
    if df.empty:
        return df
    
    # Define key columns for deduplication
    key_columns = ["Invoice number", "Entry type", "Amount settled"]
    
    # Check if key columns exist in the DataFrame
    existing_key_columns = [col for col in key_columns if col in df.columns]
    
    # If we have some key columns, deduplicate
    if existing_key_columns:
        # Drop exact duplicates
        df = df.drop_duplicates(subset=existing_key_columns)
        
        # For rows with same invoice number but different amounts,
        # keep the one with more non-null values
        if "Invoice number" in df.columns:
            # Count non-null values per row
            df["info_count"] = df.notna().sum(axis=1)
            
            # Sort by count (descending) and keep first occurrence of each invoice
            df = df.sort_values("info_count", ascending=False)
            df = df.drop_duplicates(subset=["Invoice number"], keep="first")
            
            # Drop the helper column
            df = df.drop(columns=["info_count"])
    
    return df
