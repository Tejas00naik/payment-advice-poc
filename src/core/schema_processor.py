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


def create_final_df(llm_items: List[Dict[str, Any]], metadata: Dict[str, str]) -> pd.DataFrame:
    """Create validated DataFrame from LLM output"""
    # Validate input
    if not llm_items:
        logging.warning("No items received for final dataframe")
        return pd.DataFrame(columns=FINAL_COLUMNS)
    
    # Check for missing fields
    missing_fields = [col for col in FINAL_COLUMNS if col not in llm_items[0]]
    if missing_fields:
        logging.warning(f"Missing fields in LLM output: {', '.join(missing_fields)}")
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
    df["Payment Advice number"] = df["Payment Advice number"].fillna(metadata.get("payment_advice_no"))
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
    """Ensure data quality before saving"""
    # Skip validation if empty DataFrame
    if df.empty:
        logging.warning("Empty DataFrame received for validation")
        return True
    
    # Add entry count validation
    entry_count = len(df)
    if entry_count < 10:
        logging.warning(f"Low entry count: {entry_count}. Expected 20+ entries")
    
    # Add null check
    null_counts = df.isnull().sum()
    high_null = null_counts[null_counts > 0]
    if not high_null.empty:
        logging.warning("Null values detected:\n" + high_null.to_string())
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
