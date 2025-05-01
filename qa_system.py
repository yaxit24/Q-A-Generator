import fitz  # PyMuPDF
import pdfplumber
import cv2
import pytesseract
from PIL import Image
import numpy as np
import pandas as pd
from transformers import pipeline
from flask import Flask, request, render_template_string, jsonify, redirect, url_for
import os
import io
import tempfile
import time
import json
from tqdm import tqdm
import logging
import sys
import subprocess
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Document Q&A Generator')
parser.add_argument('--port', type=int, default=5001, help='Port to run the web server on')
args = parser.parse_args()

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Check if Ollama is available
def is_ollama_available():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False

# Initialize NLP model for question generation
def initialize_model(use_ollama=True, model_name="valhalla/t5-small-e2e-qg"):
    # Try to use Ollama by default
    if is_ollama_available() and use_ollama:
        logger.info("Using Ollama for question generation")
        try:
            import ollama
            return "ollama", "llama3.2"  # Using llama3.2 which is available on your system
        except ImportError:
            logger.warning("Ollama Python package not found")
    
    # Only if Ollama is not available or explicitly requested not to use it
    logger.info(f"Using transformers pipeline with model: {model_name}")
    try:
        return "transformers", pipeline("text2text-generation", model=model_name)
    except Exception as e:
        logger.error(f"Error initializing transformers model: {str(e)}")
        logger.info("Falling back to simple question generation")
        return "simple", None

# Step 1: Document Ingestion and Parsing
def extract_text_and_images(pdf_path):
    try:
        logger.info(f"Extracting text and images from {pdf_path}")
        doc = fitz.open(pdf_path)
        text = ""
        images = []
        
        for page_num, page in enumerate(doc):
            logger.info(f"Processing page {page_num+1}/{len(doc)}")
            text += page.get_text("text") + "\n"
            
            # Extract images
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Store image data
                images.append({
                    'page': page_num + 1,
                    'index': img_index,
                    'data': image_bytes,
                    'ext': base_image["ext"]
                })
        
        doc.close()
        return text, images
    except Exception as e:
        logger.error(f"Error extracting text and images: {str(e)}")
        return "", []

# Step 2: Layout Analysis and Table Extraction
def extract_layout_and_tables(pdf_path):
    try:
        logger.info(f"Extracting layout and tables from {pdf_path}")
        with pdfplumber.open(pdf_path) as pdf:
            layout_text = ""
            all_tables = []
            
            for page_num, page in enumerate(pdf.pages):
                logger.info(f"Processing page layout {page_num+1}/{len(pdf.pages)}")
                # Extract text with layout
                page_text = page.extract_text(layout=True)
                if page_text:
                    layout_text += page_text + "\n"
                
                # Extract tables
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        if table:  # Check if table is not empty
                            all_tables.append({
                                'page': page_num + 1,
                                'data': table
                            })
            
            return layout_text, all_tables
    except Exception as e:
        logger.error(f"Error extracting layout and tables: {str(e)}")
        return "", []

# Step 3: OCR for Images
def ocr_image(image_data):
    try:
        # Convert image data to OpenCV format
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Preprocess the image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Perform OCR
        text = pytesseract.image_to_string(thresh)
        return text
    except Exception as e:
        logger.error(f"Error performing OCR: {str(e)}")
        return ""

# Step 4: Table Processing
def process_tables(tables):
    processed_tables = []
    
    for table_info in tables:
        try:
            table = table_info['data']
            page = table_info['page']
            
            # Convert table to DataFrame
            if table and len(table) > 0:
                # Use first row as header if possible
                if len(table) > 1:
                    df = pd.DataFrame(table[1:], columns=table[0])
                else:
                    df = pd.DataFrame(table)
                
                # Convert DataFrame to string representation
                table_string = df.to_string(index=False)
                
                processed_tables.append({
                    'page': page,
                    'table_text': table_string
                })
        except Exception as e:
            logger.error(f"Error processing table: {str(e)}")
    
    return processed_tables

# Simple question generation function that doesn't rely on external models
def generate_simple_questions(text):
    # Split text into sentences and generate basic questions
    sentences = text.split('.')
    qa_pairs = []
    
    # Group 3-5 related sentences together for more context
    contextual_chunks = []
    current_chunk = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 30:  # Only consider substantial sentences
            current_chunk.append(sentence)
            
            # Once we have 3-5 sentences, create a chunk
            if len(current_chunk) >= 3:
                contextual_chunks.append(" ".join(current_chunk))
                # Keep the last sentence for overlap
                current_chunk = [current_chunk[-1]] if current_chunk else []
    
    # Add any remaining sentences as a chunk
    if current_chunk:
        contextual_chunks.append(" ".join(current_chunk))
    
    # Generate questions for each contextual chunk
    for chunk in contextual_chunks:
        if len(chunk.split()) < 15:  # Skip very short chunks
            continue
            
        words = chunk.split()
        
        # Create different types of detailed questions based on the content
        questions = []
        
        # Extract key nouns from the first 10 words to use as topics
        first_part = " ".join(words[:10])
        # Find topics (likely nouns) - look for capitalized words or words longer than 5 chars
        potential_topics = [w for w in words[:15] if len(w) > 5 or (w[0].isupper() and len(w) > 3)]
        topics = potential_topics[:2]  # Take at most 2 topics
        
        if topics:
            topic_text = " and ".join(topics)
            questions.append(f"What is explained about {topic_text} in this section?")
            
            if "policy" in chunk.lower() or "procedure" in chunk.lower() or "guideline" in chunk.lower():
                questions.append(f"What are the key requirements related to {topic_text}?")
            
            if "should" in chunk.lower() or "must" in chunk.lower() or "require" in chunk.lower():
                questions.append(f"What are the obligations regarding {topic_text}?")
                
            # More intelligent contextual questions
            if any(term in chunk.lower() for term in ["purpose", "goal", "objective", "aim"]):
                questions.append(f"What is the purpose described in this section?")
                
            if any(term in chunk.lower() for term in ["responsible", "responsibility", "accountable"]):
                questions.append(f"Who is responsible for the activities described here?")
                
            if any(term in chunk.lower() for term in ["prohibited", "not allowed", "forbidden", "restricted"]):
                questions.append(f"What activities are prohibited according to this text?")
        else:
            # Fallback questions if no good topics found
            questions.append("What is the main point of this section?")
            questions.append("What information is being conveyed in this text?")
        
        # Add questions that pass quality check
        for question in questions:
            if quality_check_question(question):
                qa_pairs.append({
                    "question": question,
                    "answer": chunk,  # Use the entire contextual chunk as the answer
                    "source": "simple-detailed"
                })
                
                # Limit to 3 questions per chunk to avoid redundancy
                if len(qa_pairs) % 3 == 0:
                    break
    
    # Ensure we don't return too many questions
    return qa_pairs[:20]

# Step 5: Q&A Generation
def generate_questions(text, model_info, max_questions=20):
    model_type, model = model_info
    
    # Split text into larger chunks for more context and detailed answers
    chunk_size = 1000  # Increased from 600
    overlap = 250      # Increased from 150
    chunks = []
    
    # Create overlapping chunks
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk.strip()) > 100:  # Only use chunks with more substantial content (increased from 50)
            chunks.append(chunk)
    
    # Limit number of chunks to process
    max_chunks = 100
    if len(chunks) > max_chunks:
        logger.info(f"Limiting to {max_chunks} chunks for processing")
        chunks = chunks[:max_chunks]
    
    qa_pairs = []
    ollama_failed = False
    
    if model_type == "transformers":
        try:
            for chunk in tqdm(chunks, desc="Generating questions"):
                if chunk.strip():
                    try:
                        # Generate questions using transformer model with higher max_length
                        questions = model(chunk, max_length=128, num_return_sequences=3)  # Increased from 64
                        
                        for q in questions:
                            question_text = q["generated_text"].strip()
                            if quality_check_question(question_text):
                                qa_pairs.append({
                                    "question": question_text,
                                    "answer": chunk.strip(),
                                    "source": "t5"
                                })
                                
                                # Limit to max questions per document
                                if len(qa_pairs) >= max_questions:
                                    break
                        
                        if len(qa_pairs) >= max_questions:
                            break
                    except Exception as e:
                        logger.error(f"Error generating questions with transformers: {str(e)}")
        except Exception as e:
            logger.error(f"Transformer pipeline failed: {str(e)}")
            # If transformer fails, we'll fall back to simple generation later
    
    elif model_type == "ollama":
        try:
            import ollama
            
            # First check if we can connect to the model by sending a test message
            try:
                test_response = ollama.chat(model=model, messages=[{"role": "user", "content": "Hello"}])
                logger.info(f"Successfully connected to Ollama model {model}")
            except Exception as e:
                logger.error(f"Failed to connect to Ollama model {model}: {str(e)}")
                ollama_failed = True
            
            if not ollama_failed:
                # Increase from 40 to 50 chunks for Ollama
                for chunk in tqdm(chunks[:50], desc="Generating questions with Ollama"):
                    if chunk.strip():
                        try:
                            # Ask Ollama to generate more detailed questions with longer answers
                            prompt = f"""Generate 3 high-quality, focused questions based on this text. 
                            The questions should:
                            1. Focus on the most important or useful information in the text
                            2. Be clear, concise, and grammatically correct
                            3. Require detailed knowledge of the text to answer
                            4. Not be overly complicated or use complex language
                            5. Cover different aspects of the information when possible
                            
                            Format each question as a complete sentence with a question mark.
                            Only output the questions, nothing else:

                            {chunk}"""
                            response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
                            
                            # Parse questions from response
                            questions = [q.strip() for q in response['message']['content'].split('\n') if '?' in q]
                            
                            for question in questions:
                                if quality_check_question(question):
                                    # Generate a more detailed answer for this question based on the chunk
                                    answer_prompt = f"""Based on this information:
                                    {chunk}
                                    
                                    Please provide a clear, comprehensive answer to this question:
                                    {question}
                                    
                                    Your answer should:
                                    1. Be factual and based only on the information provided
                                    2. Be complete and thorough
                                    3. Be well-organized and easy to understand
                                    4. Use natural language without unnecessary technical terms
                                    """
                                    
                                    try:
                                        answer_response = ollama.chat(model=model, messages=[{"role": "user", "content": answer_prompt}])
                                        detailed_answer = answer_response['message']['content'].strip()
                                        
                                        # Combine the original context with the generated answer
                                        final_answer = detailed_answer if len(detailed_answer) > 100 else chunk.strip()
                                        
                                        qa_pairs.append({
                                            "question": question,
                                            "answer": final_answer,
                                            "source": "ollama-detailed"
                                        })
                                    except Exception as e:
                                        # Fall back to using the chunk as the answer
                                        logger.error(f"Error generating detailed answer: {str(e)}")
                                        qa_pairs.append({
                                            "question": question,
                                            "answer": chunk.strip(),
                                            "source": "ollama"
                                        })
                                    
                                    # Limit to max questions
                                    if len(qa_pairs) >= max_questions:
                                        break
                            
                            if len(qa_pairs) >= max_questions:
                                break
                        except Exception as e:
                            logger.error(f"Error generating questions with Ollama: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to use Ollama: {str(e)}")
            ollama_failed = True
    
    # If we still don't have any qa_pairs, use simple generation as fallback
    if len(qa_pairs) == 0 or model_type == "simple" or ollama_failed:
        logger.info("Using simple question generation")
        # Use the simple question generation method
        for chunk in tqdm(chunks[:50], desc="Generating simple questions"):
            simple_qa_pairs = generate_simple_questions(chunk)
            qa_pairs.extend(simple_qa_pairs)
            
            if len(qa_pairs) >= max_questions:
                qa_pairs = qa_pairs[:max_questions]
                break
    
    # Deduplicate questions
    unique_qa_pairs = []
    seen_questions = set()
    
    for qa in qa_pairs:
        question_key = qa["question"].lower()
        if question_key not in seen_questions:
            seen_questions.add(question_key)
            unique_qa_pairs.append(qa)
    
    return unique_qa_pairs

# Step 6: Quality Check
def quality_check_question(question):
    # Check if the question ends with a question mark
    if not question.endswith('?'):
        return False
    
    # Check if the question has a minimum length
    if len(question.split()) < 6:  # Increased from 5 to require more substantial questions
        return False
    
    # Check if the question has a maximum length (increased for more detailed questions)
    if len(question.split()) > 25:  # Reduced from 30 to make questions more focused
        return False
    
    # Check if the question starts with a question word
    question_starters = ['what', 'who', 'where', 'when', 'why', 'how', 'which', 'can', 'does', 'do', 'is', 'are', 'could', 'would', 'should', 'explain']
    if not any(question.lower().startswith(starter) for starter in question_starters):
        return False
    
    # Avoid questions that just echo back large portions of text
    words = question.split()
    question_length = len(words)
    
    # Check for ellipsis which may indicate truncated text
    if "..." in question:
        position = words.index("...")
        if position < question_length - 2:  # If ellipsis is not near the end
            return False
    
    # Avoid questions that look like they're cutting off sentences
    if any(x in question for x in ["according to the passage", "as described in", "Can you provide a detailed explanation of"]):
        # These phrases often appear in the poor quality template questions
        if question.count("?") > 1 or question.count("...") > 0:
            return False
    
    # Reject questions with very unnatural phrases (these appeared in your bad examples)
    bad_phrases = [
        "significance of", 
        "implications of", 
        "what is the significance",
        "can you provide a detailed explanation"
    ]
    
    if any(phrase in question.lower() for phrase in bad_phrases):
        return False
    
    return True

# Step 7: Process Document
def process_document(file_path, use_ollama=False):
    start_time = time.time()
    logger.info(f"Starting document processing for {file_path}")
    
    try:
        # Initialize the model
        model_info = initialize_model(use_ollama)
        
        # Extract text, images, and tables
        basic_text, images = extract_text_and_images(file_path)
        layout_text, tables = extract_layout_and_tables(file_path)
        
        # Process tables
        processed_tables = process_tables(tables)
        table_text = "\n".join([t['table_text'] for t in processed_tables])
        
        # OCR on images (optional)
        if images:
            logger.info(f"Processing {len(images)} images with OCR")
            ocr_text = ""
            for img in tqdm(images[:10], desc="OCR Processing"):  # Limit to first 10 images
                img_text = ocr_image(img['data'])
                if img_text.strip():
                    ocr_text += f"Image on page {img['page']}:\n{img_text}\n\n"
        else:
            ocr_text = ""
        
        # Combine all text
        full_text = (basic_text + "\n" + layout_text + "\n" + 
                     (ocr_text if ocr_text else "") + "\n" + 
                     (table_text if table_text else ""))
        
        # Generate Q&A pairs
        logger.info("Generating Q&A pairs")
        qa_pairs = generate_questions(full_text, model_info)
        
        processing_time = time.time() - start_time
        logger.info(f"Document processing completed in {processing_time:.2f} seconds")
        
        return {
            "qa_pairs": qa_pairs,
            "stats": {
                "processing_time": processing_time,
                "text_length": len(full_text),
                "num_images": len(images),
                "num_tables": len(processed_tables),
                "num_qa_pairs": len(qa_pairs)
            }
        }
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        return {"error": str(e)}

# HTML Templates
UPLOAD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Document Q&A Generator</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .upload-container {
            border: 2px dashed #ccc;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            text-align: center;
        }
        .upload-container:hover {
            border-color: #3498db;
        }
        input[type="file"] {
            display: none;
        }
        .upload-btn {
            background-color: #3498db;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            display: inline-block;
        }
        .submit-btn {
            background-color: #2ecc71;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .options {
            margin: 20px 0;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }
        .file-name {
            margin-top: 10px;
            font-weight: bold;
        }
        @keyframes spinner {
            to {transform: rotate(360deg);}
        }
        .spinner {
            display: none;
            width: 30px;
            height: 30px;
            border: 4px solid rgba(0, 0, 0, 0.1);
            border-left-color: #3498db;
            border-radius: 50%;
            animation: spinner 1s linear infinite;
            margin: 20px auto;
        }
        .error-message {
            background-color: #f8d7da;
            color: #721c24;
            padding: 10px;
            border-radius: 5px;
            margin: 20px 0;
            display: {{ 'block' if error else 'none' }};
        }
    </style>
</head>
<body>
    <h1>Document Q&A Generator</h1>
    <p>Upload a PDF document to automatically generate questions and answers from its content.</p>
    
    <div class="error-message" id="errorMessage">
        {{ error }}
    </div>
    
    <form method="post" enctype="multipart/form-data" id="uploadForm">
        <div class="upload-container">
            <label for="file" class="upload-btn">Choose PDF File</label>
            <input type="file" name="file" id="file" accept=".pdf" onchange="updateFileName()">
            <div class="file-name" id="fileName"></div>
        </div>
        
        <div class="options">
            <h3>Processing Options</h3>
            <input type="checkbox" id="useOllama" name="use_ollama" checked>
            <label for="useOllama">Use Ollama (if available) for better quality questions</label>
        </div>
        
        <div style="text-align: center;">
            <button type="submit" class="submit-btn" onclick="showSpinner()">Generate Q&A Pairs</button>
            <div class="spinner" id="spinner"></div>
        </div>
    </form>
    
    <script>
        function updateFileName() {
            const fileInput = document.getElementById('file');
            const fileNameDiv = document.getElementById('fileName');
            if (fileInput.files.length > 0) {
                fileNameDiv.textContent = "Selected file: " + fileInput.files[0].name;
            } else {
                fileNameDiv.textContent = "";
            }
        }
        
        function showSpinner() {
            if (document.getElementById('file').files.length > 0) {
                document.getElementById('spinner').style.display = 'block';
            }
        }
    </script>
</body>
</html>
"""

RESULTS_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Q&A Results</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        h1, h2 {
            color: #333;
        }
        .qa-pair {
            background-color: #f9f9f9;
            padding: 20px;
            margin-bottom: 25px;
            border-radius: 8px;
            border-left: 5px solid #3498db;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .question {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 18px;
            line-height: 1.4;
        }
        .answer {
            color: #333;
            overflow-wrap: break-word;
            white-space: pre-line;
            padding: 10px;
            background-color: #fff;
            border-radius: 5px;
            border-left: 3px solid #2ecc71;
            font-size: 15px;
            line-height: 1.5;
            max-height: 400px;
            overflow-y: auto;
        }
        .stats {
            background-color: #e8f4f8;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .back-btn {
            display: inline-block;
            margin-top: 20px;
            background-color: #3498db;
            color: white;
            padding: 12px 25px;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
            transition: background-color 0.3s;
        }
        .back-btn:hover {
            background-color: #2980b9;
        }
        .highlight {
            font-weight: bold;
            color: #e74c3c;
        }
        .download-btn {
            display: inline-block;
            margin-left: 15px;
            background-color: #2ecc71;
            color: white;
            padding: 12px 25px;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
            transition: background-color 0.3s;
        }
        .download-btn:hover {
            background-color: #27ae60;
        }
        .source-tag {
            display: inline-block;
            font-size: 12px;
            background-color: #f1c40f;
            color: #333;
            padding: 3px 8px;
            border-radius: 10px;
            margin-left: 10px;
            vertical-align: middle;
        }
        .qa-controls {
            text-align: center;
            margin-bottom: 20px;
        }
        .filter-btn {
            margin: 0 5px;
            padding: 8px 15px;
            background-color: #ddd;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .filter-btn:hover, .filter-btn.active {
            background-color: #3498db;
            color: white;
        }
    </style>
</head>
<body>
    <h1>Generated Q&A Pairs</h1>
    
    <div class="stats">
        <h2>Processing Statistics</h2>
        <p><strong>Processing Time:</strong> {{ stats.processing_time|round(2) }} seconds</p>
        <p><strong>Document Size:</strong> {{ (stats.text_length / 1000)|round(1) }} KB of text</p>
        <p><strong>Images Processed:</strong> {{ stats.num_images }}</p>
        <p><strong>Tables Extracted:</strong> {{ stats.num_tables }}</p>
        <p><strong>Q&A Pairs Generated:</strong> <span class="highlight">{{ stats.num_qa_pairs }}</span></p>
    </div>
    
    <div class="qa-controls">
        <button class="filter-btn active" onclick="filterQA('all')">All</button>
        <button class="filter-btn" onclick="filterQA('ollama')">Ollama Generated</button>
    </div>
    
    {% if qa_pairs %}
        {% for qa in qa_pairs %}
        <div class="qa-pair" data-source="{{ qa.source }}">
            <div class="question">Q: {{ qa.question }}</div>
            <div class="answer">{{ qa.answer }}</div>
        </div>
        {% endfor %}
    {% else %}
        <p>No Q&A pairs were generated. The document might not contain suitable content.</p>
    {% endif %}
    
    <a href="/" class="back-btn">Process Another Document</a>
    <a href="/download" class="download-btn">Download Results (JSON)</a>
    
    <script>
        function filterQA(filter) {
            // Update active button
            document.querySelectorAll('.filter-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            event.target.classList.add('active');
            
            // Filter the QA pairs
            document.querySelectorAll('.qa-pair').forEach(pair => {
                const source = pair.getAttribute('data-source');
                if (filter === 'all') {
                    pair.style.display = 'block';
                } else if (filter === 'ollama' && source.includes('ollama')) {
                    pair.style.display = 'block';
                } else {
                    pair.style.display = 'none';
                }
            });
        }
    </script>
</body>
</html>
"""

# Step 8: Flask UI
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['SECRET_KEY'] = os.urandom(24)
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Store results in memory for this session
session_results = {}

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if there's a file in the request
        if 'file' not in request.files:
            logger.error("No file part in the request")
            return render_template_string(UPLOAD_HTML, error="No file selected")
        
        file = request.files['file']
        
        # Check if user submitted an empty form
        if file.filename == '':
            logger.error("Empty filename submitted")
            return render_template_string(UPLOAD_HTML, error="No file selected")
        
        # Check if the file is a PDF
        if file and file.filename.endswith('.pdf'):
            try:
                # Save the file to the upload folder
                filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filename)
                logger.info(f"File saved to {filename}")
                
                # Process the saved file
                use_ollama = 'use_ollama' in request.form
                results = process_document(filename, use_ollama)
                
                if 'error' in results:
                    logger.error(f"Error processing document: {results['error']}")
                    return render_template_string(UPLOAD_HTML, error=f"Error processing document: {results['error']}")
                
                # Store results for download
                session_id = os.urandom(16).hex()
                session_results[session_id] = results
                
                # Set session ID in cookie
                resp = redirect(url_for('show_results', session_id=session_id))
                resp.set_cookie('session_id', session_id)
                return resp
            except Exception as e:
                logger.error(f"Unexpected error during file upload: {str(e)}")
                return render_template_string(UPLOAD_HTML, error=f"Unexpected error: {str(e)}")
        else:
            logger.error(f"Invalid file type: {file.filename}")
            return render_template_string(UPLOAD_HTML, error="Please upload a PDF file")
    
    return render_template_string(UPLOAD_HTML)

@app.route('/results/<session_id>', methods=['GET'])
def show_results(session_id):
    if session_id in session_results:
        results = session_results[session_id]
        return render_template_string(RESULTS_HTML, qa_pairs=results['qa_pairs'], stats=results['stats'])
    else:
        return redirect(url_for('upload_file'))

@app.route('/download', methods=['GET'])
def download_results():
    session_id = request.cookies.get('session_id')
    if session_id and session_id in session_results:
        results = session_results[session_id]
        
        # Prepare JSON data
        download_data = {
            "qa_pairs": results['qa_pairs'],
            "stats": results['stats'],
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Convert to JSON
        json_data = json.dumps(download_data, indent=2)
        
        # Create a response with JSON file
        output = io.BytesIO(json_data.encode('utf-8'))
        output.seek(0)
        
        return jsonify(download_data)
    
    return redirect(url_for('upload_file'))

if __name__ == '__main__':
    print("Document Q&A System initializing...")
    print(f"Ollama available: {is_ollama_available()}")
    print("Starting web server at http://localhost:" + str(args.port))
    app.run(debug=True, host='0.0.0.0', port=args.port) 