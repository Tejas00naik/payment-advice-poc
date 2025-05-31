#!/usr/bin/env python3
"""
Main entry point for the Automated Payment Advice Extraction System.
"""
import os
import pandas as pd
from dotenv import load_dotenv

from email_processor import EmailProcessor
from document_processor import DocumentProcessor
from data_extractor import DataExtractor
from utils import setup_logging

# Load environment variables
load_dotenv()

def main():
    """Main function to orchestrate the extraction process."""
    logger = setup_logging()
    logger.info("Starting Automated Payment Advice Extraction System")
    
    # Initialize components
    email_proc = EmailProcessor()
    doc_proc = DocumentProcessor()
    data_extractor = DataExtractor()
    
    # Process emails and get attachments
    logger.info("Processing emails")
    emails = email_proc.fetch_emails()
    
    all_data = []
    
    for email in emails:
        # Extract text from email body
        email_text = email_proc.extract_email_text(email)
        email_data = data_extractor.extract_from_text(email_text)
        
        if email_data:
            email_data['source'] = 'email_body'
            all_data.append(email_data)
        
        # Process attachments
        attachments = email_proc.extract_attachments(email)
        
        for attachment in attachments:
            file_path = attachment['path']
            file_type = attachment['type']
            
            # Extract text from attachment
            if file_type == 'pdf':
                text = doc_proc.process_pdf(file_path)
            elif file_type in ['png', 'jpg', 'jpeg', 'tiff']:
                text = doc_proc.process_image(file_path)
            else:
                text = doc_proc.process_text_file(file_path)
            
            # Extract structured data from text
            attachment_data = data_extractor.extract_from_text(text)
            
            if attachment_data:
                attachment_data['source'] = f'attachment_{file_type}'
                all_data.append(attachment_data)
    
    # Combine all extracted data into a DataFrame
    if all_data:
        df = pd.DataFrame(all_data)
        logger.info(f"Extracted {len(df)} records")
        
        # Save to CSV
        output_path = os.path.join(os.getcwd(), 'extracted_data.csv')
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")
        
        return df
    else:
        logger.warning("No data was extracted")
        return pd.DataFrame()

if __name__ == "__main__":
    main()
