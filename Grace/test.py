import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
import faiss
import numpy as np
from huggingface_hub import login
from typing import List, Dict

class AutoQuestionGenerator:
    """
    Implementation of the Automatic Question Generation (AQG) module as described in section 3.1
    """
    def __init__(self, model_name: str, access_token: str):
        # Login to Hugging Face
        login(token=access_token)
        
        # Initialize model and tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
        
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token)
        
        print("Creating pipeline...")
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        self.llm = HuggingFacePipeline(pipeline=self.pipeline)
        
        # Initialize question types
        self.question_types = ["What", "Where", "Who", "When", "Why", "How"]
        
        # Initialize prompts
        self.background_prompt = PromptTemplate(
            template="""System: You are a professional news writer specializing in comprehensive background information.
            
            User: Generate detailed background information for the topic: {topic}
            
            Output should:
            1. Cover key aspects and context
            2. Include relevant details and developments
            3. Provide balanced information
            4. Be factual and objective
            
            Background:""",
            input_variables=["topic"]
        )
        
        self.question_prompt = PromptTemplate(
            template="""System: You are an expert at generating insightful questions in the 5W1H format.
            
            User: Generate {num_questions} questions that start with {question_type} based on this background:
            {background}
            
            Requirements:
            1. Each question must start with {question_type}
            2. Questions should be directly relevant to the background
            3. Questions should seek specific information
            4. Use complete, well-formed sentences
            
            Questions:""",
            input_variables=["num_questions", "question_type", "background"]
        )

    def generate_background(self, topic: str) -> str:
        """Step 1-2: Generate background information"""
        print(f"\nGenerating background for topic: {topic}")
        try:
            response = self.llm.invoke(self.background_prompt.format(topic=topic))
            print("Background generated successfully")
            return response
        except Exception as e:
            print(f"Error generating background: {e}")
            raise

    def generate_questions(self, background: str) -> Dict[str, List[str]]:
        """Step 3-4: Generate initial questions for each type"""
        print("\nGenerating initial questions...")
        questions = {}
        
        for qtype in self.question_types:
            try:
                response = self.llm.invoke(
                    self.question_prompt.format(
                        num_questions=5,
                        question_type=qtype,
                        background=background
                    )
                )
                # Parse response into list of questions
                questions[qtype] = [q.strip() for q in response.split('\n') if q.strip().startswith(qtype)]
                print(f"Generated {len(questions[qtype])} questions for {qtype}")
            except Exception as e:
                print(f"Error generating questions for {qtype}: {e}")
                questions[qtype] = []
                
        return questions

    def select_questions(self, questions: Dict[str, List[str]], background: str, k: int = 2) -> List[str]:
        """Step 5-6: Select most relevant questions using FAISS"""
        print("\nSelecting most relevant questions...")
        
        # Initialize FAISS index
        dimension = 384  # text2vec dimension
        index = faiss.IndexFlatL2(dimension)
        
        # Convert background to vector (simplified representation)
        background_vector = np.random.rand(1, dimension).astype('float32')
        index.add(background_vector)
        
        selected_questions = []
        for qtype, qlist in questions.items():
            if not qlist:
                continue
                
            # Convert questions to vectors (simplified)
            q_vectors = np.random.rand(len(qlist), dimension).astype('float32')
            
            # Get top k similar questions
            D, I = index.search(q_vectors, k)
            
            # Select questions
            for idx in I[0][:k]:
                if idx < len(qlist):  # Safety check
                    selected_questions.append(qlist[idx])
                    
        print(f"Selected {len(selected_questions)} questions")
        return selected_questions

    def process_topic(self, topic: str) -> List[str]:
        """Main method to process a topic and generate questions"""
        try:
            # Generate background
            background = self.generate_background(topic)
            
            # Generate initial questions
            questions = self.generate_questions(background)
            
            # Select final questions
            final_questions = self.select_questions(questions, background)
            
            return final_questions
            
        except Exception as e:
            print(f"Error processing topic: {e}")
            return []

def main():
    # Configuration
    MODEL_NAME = "meta-llama/Llama-3.2-3B"
    ACCESS_TOKEN = "token"
    
    # Initialize generator
    generator = AutoQuestionGenerator(MODEL_NAME, ACCESS_TOKEN)
    
    # Test topics
    topics = [
        "2024 Election"
    ]
    
    # Process each topic
    for topic in topics:
        print(f"\nProcessing topic: {topic}")
        questions = generator.process_topic(topic)
        
        print(f"\nGenerated questions for {topic}:")
        for i, question in enumerate(questions, 1):
            print(f"{i}. {question}")
        print("-" * 50)

if __name__ == "__main__":
    main()