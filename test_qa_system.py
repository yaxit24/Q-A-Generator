import sys
import unittest
import tempfile
import os
from io import BytesIO
import time

# Import the qa_system module
try:
    from qa_system import (
        extract_text_and_images,
        extract_layout_and_tables,
        ocr_image,
        process_tables,
        quality_check_question,
        initialize_model,
        generate_questions
    )
except ImportError:
    print("Error: Could not import qa_system module. Make sure qa_system.py is in the current directory.")
    sys.exit(1)

class TestQASystem(unittest.TestCase):
    
    def test_quality_check(self):
        """Test the quality check function"""
        # Test valid questions
        valid_questions = [
            "What is the capital of France?",
            "How does photosynthesis work?",
            "When was the Declaration of Independence signed?",
            "Why is the sky blue?",
            "Where is the Great Barrier Reef located?"
        ]
        
        # Test invalid questions
        invalid_questions = [
            "What",  # Too short
            "This is not a question",  # No question mark
            "A? B? C?",  # Too short
            "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua ut enim ad minim veniam?",  # Too long
            "yesterday the weather was nice?",  # Doesn't start with question word
        ]
        
        # Check valid questions
        for q in valid_questions:
            self.assertTrue(quality_check_question(q), f"Should accept valid question: {q}")
        
        # Check invalid questions
        for q in invalid_questions:
            self.assertFalse(quality_check_question(q), f"Should reject invalid question: {q}")
    
    def test_model_initialization(self):
        """Test model initialization"""
        try:
            model_type, model = initialize_model(use_ollama=False)
            self.assertEqual(model_type, "transformers", "Model type should be 'transformers'")
            self.assertIsNotNone(model, "Model should not be None")
        except Exception as e:
            self.fail(f"Model initialization failed with error: {str(e)}")
    
    def test_question_generation(self):
        """Test question generation with a sample text"""
        # Sample text
        sample_text = """
        Artificial intelligence (AI) is intelligence demonstrated by machines, 
        as opposed to intelligence displayed by animals including humans. 
        AI research has been defined as the field of study of intelligent agents, 
        which refers to any system that perceives its environment and takes actions 
        that maximize its chance of achieving its goals.
        """
        
        # Initialize model
        model_info = initialize_model(use_ollama=False)
        
        # Generate questions
        questions = generate_questions(sample_text, model_info, max_questions=2)
        
        # Check if questions were generated
        self.assertGreater(len(questions), 0, "Should generate at least one question")
        
        # Check if questions have the correct format
        for qa in questions:
            self.assertIn("question", qa, "Each QA pair should have a 'question' field")
            self.assertIn("answer", qa, "Each QA pair should have an 'answer' field")
            self.assertTrue(qa["question"].endswith("?"), "Questions should end with a question mark")
    
    def test_create_temp_pdf(self):
        """Create a temporary PDF file for testing"""
        from reportlab.pdfgen import canvas
        
        # Create a temporary PDF file
        pdf_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        pdf_path = pdf_file.name
        pdf_file.close()
        
        # Create a simple PDF with reportlab
        c = canvas.Canvas(pdf_path)
        c.drawString(100, 750, "Test PDF Document")
        c.drawString(100, 700, "This is a test document for the QA system.")
        c.save()
        
        # Test if the file exists
        self.assertTrue(os.path.exists(pdf_path), "Temporary PDF file should exist")
        
        # Clean up
        os.unlink(pdf_path)

if __name__ == "__main__":
    print("Running QA System Tests...")
    unittest.main() 