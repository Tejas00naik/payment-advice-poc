#!/usr/bin/env python3
"""
Schema Processor for Payment Advice Extraction System.
Handles data validation, schema enforcement, and final DataFrame creation.
"""
import pandas as pd
from typing import List, Dict, Any
import datetime

# Define final schema columns
FINAL_COLUMNS = [
    "Payment date", "Payment advice number", "Invoice number",
    "Invoice date", "Customer name", "Customer id (SAP id)",
    "UTR", "Doc type", "Doc number", "Amount paid",
    "TDS deducted", "Currency", "PA link"
]


def create_final_df(llm_items: List[Dict[str, Any]], metadata: Dict[str, str]) -> pd.DataFrame:
    """
    Create validated DataFrame from LLM output
    
    Args:
        llm_items: List of dictionaries with structured payment advice data
        metadata: Dictionary of extracted metadata from the document
        
    Returns:
        DataFrame with validated and standardized data
    """
    # Create DataFrame from items
    if not llm_items:
        # Return empty DataFrame with correct columns if no items found
        return pd.DataFrame(columns=FINAL_COLUMNS)
    
    df = pd.DataFrame(llm_items)
    
    # Ensure all columns exist
    for col in FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    
    # Post-processing: fill in metadata
    df["Payment date"] = df["Payment date"].fillna(metadata.get("payment_date"))
    df["Payment advice number"] = df["Payment advice number"].fillna(metadata.get("payment_advice_no"))
    df["Customer name"] = df["Customer name"].fillna(metadata.get("customer_name"))
    df["Currency"] = df["Currency"].fillna(metadata.get("currency"))
    
    # Type conversion for numeric fields
    for col in ["Amount paid", "TDS deducted"]:
        if col in df.columns:
            # Handle different formats (commas, currency symbols)
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace('[^0-9.-]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Date formatting
    for date_col in ["Payment date", "Invoice date"]:
        if date_col in df.columns and df[date_col].notna().any():
            # Attempt to convert the column to datetime, specifying a common format.
            # errors='coerce' will turn unparseable dates into NaT.
            # LLM is prompted for YYYY-MM-DD for Invoice Date.
            df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m-%d', errors='coerce')
            # If parsing with specific format fails for all and there was data, log it.
            # This check helps if the primary format assumption is wrong for a particular file.
            if df[date_col].isna().all() and df[date_col].notna().any():
                 logging.warning(f"All values in date column '{date_col}' became NaT after parsing with format '%Y-%m-%d'. Trying with infer_datetime_format.")
                 # Fallback to inferring if the primary format fails for all entries
                 df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True, errors='coerce')
                 if df[date_col].isna().all() and df[date_col].notna().any():
                    logging.error(f"All values in date column '{date_col}' still NaT after inferring format. Original data: {df[date_col].unique()[:5]}")
    
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
        (df["Payment advice number"].nunique() <= 1 or df["Payment advice number"].isnull().all(), 
         "Multiple payment advice numbers"),
        
        # Check for negative total amount (could be valid in some cases)
        (not (df["Amount paid"].sum() < 0), "Negative total amount"),
        
        # Check for null values in key fields
        ((df["Amount paid"].notnull()).any(), "No valid amounts found")
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
    key_columns = ["Invoice number", "Doc type", "Amount paid"]
    
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
