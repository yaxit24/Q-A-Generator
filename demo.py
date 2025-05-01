"""
Document Q&A System Demo

This script demonstrates how to use the Document Q&A system programmatically
to generate question-answer pairs from PDF documents.
"""

import os
import sys
import time
from pprint import pprint

# Import the QA system functions
from qa_system import (
    extract_text_and_images,
    extract_layout_and_tables,
    process_tables,
    initialize_model,
    generate_questions,
    quality_check_question,
    is_ollama_available
)

def main():
    print("Document Q&A System Demo")
    print("=" * 50)
    
    # Check Ollama availability
    print(f"Ollama available: {is_ollama_available()}")
    print()
    
    # Initialize the model
    print("Initializing the question generation model...")
    use_ollama = False  # Set to True to use Ollama if available
    
    start_time = time.time()
    model_type, model = initialize_model(use_ollama=use_ollama)
    init_time = time.time() - start_time
    
    print(f"Model initialized in {init_time:.2f} seconds")
    print(f"Model type: {model_type}")
    print()
    
    # Generate or use a sample PDF
    pdf_path = "example.pdf"
    
    if not os.path.exists(pdf_path):
        try:
            from reportlab.pdfgen import canvas
            print("Creating a sample PDF document...")
            c = canvas.Canvas(pdf_path)
            c.drawString(100, 750, "Sample PDF Document")
            c.drawString(100, 720, "This is a sample document generated for the Document Q&A system demo.")
            c.drawString(100, 690, "Artificial intelligence (AI) is intelligence demonstrated by machines,")
            c.drawString(100, 670, "as opposed to intelligence displayed by animals including humans.")
            c.drawString(100, 650, "AI research has been defined as the field of study of intelligent agents,")
            c.drawString(100, 630, "which refers to any system that perceives its environment and takes actions")
            c.drawString(100, 610, "that maximize its chance of achieving its goals.")
            c.save()
            print(f"Generated sample PDF: {pdf_path}")
        except ImportError:
            print("ReportLab is not installed. Please provide your own PDF file or install reportlab.")
            return
    else:
        print(f"Using existing PDF: {pdf_path}")
    
    print()
    
    # Extract text and images
    print("Extracting text and images...")
    start_time = time.time()
    basic_text, images = extract_text_and_images(pdf_path)
    text_time = time.time() - start_time
    
    print(f"Text extraction completed in {text_time:.2f} seconds")
    print(f"Extracted {len(basic_text)} characters of text")
    print(f"Extracted {len(images)} images")
    
    # Preview the extracted text
    print("\nText preview:")
    print(basic_text[:200] + "..." if len(basic_text) > 200 else basic_text)
    print()
    
    # Extract layout and tables
    print("Extracting layout and tables...")
    start_time = time.time()
    layout_text, tables = extract_layout_and_tables(pdf_path)
    layout_time = time.time() - start_time
    
    print(f"Layout extraction completed in {layout_time:.2f} seconds")
    print(f"Extracted {len(layout_text)} characters of layout text")
    print(f"Extracted {len(tables)} tables")
    
    # Process tables (if any)
    if tables:
        processed_tables = process_tables(tables)
        print(f"Processed {len(processed_tables)} tables")
        
        # Preview a table
        if processed_tables:
            print("\nTable preview:")
            print(processed_tables[0]['table_text'])
    
    print()
    
    # Generate questions
    print("Generating questions...")
    full_text = basic_text + "\n" + layout_text
    
    start_time = time.time()
    qa_pairs = generate_questions(full_text, (model_type, model), max_questions=5)
    gen_time = time.time() - start_time
    
    print(f"Question generation completed in {gen_time:.2f} seconds")
    print(f"Generated {len(qa_pairs)} question-answer pairs")
    print()
    
    # Display generated Q&A pairs
    print("Generated Q&A Pairs:")
    print("=" * 50)
    
    for i, qa in enumerate(qa_pairs):
        print(f"Q{i+1}: {qa['question']}")
        print(f"A: {qa['answer'][:100]}..." if len(qa['answer']) > 100 else f"A: {qa['answer']}")
        print()
    
    print("=" * 50)
    print("\nDemo completed!")
    print("For more information, please refer to the README.md file in the repository.")

if __name__ == "__main__":
    main() 