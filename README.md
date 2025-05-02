# Fast Document Q&A Generator

A high-performance tool for automatically generating question and answer pairs from PDF documents. This application combines fast document processing, optical character recognition (OCR), and natural language processing to create meaningful Q&A pairs from any PDF content.

## Features

- **Efficient PDF Processing**: Extract text, tables, and images from PDF documents
- **OCR Support**: Recognize text in images within documents
- **Table Extraction**: Process tabular data from documents
- **Automatic Q&A Generation**: Generate relevant questions and answers using AI
- **Quality Filtering**: Ensure only high-quality Q&A pairs are produced
- **Multiple Model Options**: Use either local models or Ollama for question generation
- **Web Interface**: Simple user interface for document upload and results display
- **JSON Export**: Download results for further use

## Quick Start

### Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd document-qa-generator
   ```

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install Tesseract OCR (required for image text extraction):
   - **macOS**: `brew install tesseract`
   - **Ubuntu/Debian**: `apt-get install tesseract-ocr`
   - **Windows**: Download installer from [here](https://github.com/UB-Mannheim/tesseract/wiki)

4. (Optional) Install Ollama for better question generation:
   - Follow instructions at [ollama.ai](https://ollama.ai)
   - Install the Python package: `pip install ollama`
   - Pull a model: `ollama pull llama2`

### Running the Application

Run the Flask web server:
```
python qa_system.py
```

Open your browser and go to [http://localhost:5000](http://localhost:5000)

## Usage

1. **Upload a Document**: Click on the upload area to select a PDF file (max 16MB)
2. **Choose Processing Options**:
   - Toggle the "Use Ollama" option if you have Ollama installed for higher-quality questions
3. **Generate Q&A Pairs**: Click the "Generate Q&A Pairs" button and wait for processing
4. **Review Results**: Examine the generated Q&A pairs and processing statistics
5. **Download**: Use the "Download Results" button to save the results as JSON

## How It Works

1. **Document Processing**: The system extracts text from PDF documents using PyMuPDF and pdfplumber
2. **Image Processing**: Images in the document are processed with OCR to extract any embedded text
3. **Table Extraction**: Tables are identified and converted to text format
4. **Text Processing**: The collected text is split into manageable chunks
5. **Question Generation**: An AI model generates questions for each text chunk
6. **Quality Checking**: Generated questions are filtered for quality and relevance
7. **Results Display**: The final Q&A pairs are presented in the web interface

## Command Line Options

The system can also be used programmatically. See `qa_system.py` for details on the following functions:

- `process_document(file, use_ollama=False)` - Process a document and generate Q&A pairs
- `extract_text_and_images(pdf_path)` - Extract raw text and images from a PDF
- `extract_layout_and_tables(pdf_path)` - Extract layout-aware text and tables
- `ocr_image(image_data)` - Perform OCR on an image
- `generate_questions(text, model_info)` - Generate questions from text

## Performance Considerations

- Processing time depends on document size, number of images, and tables
- OCR operations are limited to the first 10 images to maintain performance
- Text chunks are limited to 20 for question generation


## Requirements

- Python 3.8+
- Tesseract OCR
- 4GB+ RAM
- (Optional) Ollama for improved question generation

## What I Learnt 

- Using Python to build, which solves real-world problems for organisations.
- Learnt about the various frameworks and libraries. 
- Practiced the implementation of OOPs.
- Learnt how to research, build in 24hrs using the AI for the SecurityPal x Nammi (NepalHacks 3.0) Hackathon.

## License

MIT License 
