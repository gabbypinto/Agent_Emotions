import faiss
import numpy as np
from typing import Dict, List
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from modules.llm_model import LLMModel


class QuestionGenerator:
    def __init__(self, llm_model: LLMModel):
        self.llm_model = llm_model
        self.question_types = ["What", "Where", "Who", "When", "Why", "How"]
        self.question_prompt = PromptTemplate(
            input_variables=["question_type", "video_description", "format_instructions", "persona"],
            template="""
            You are an AI with the persona of {persona}, generating structured JSON output. 
            Generate exactly three questions of type "{question_type}" based on the following video description:

            Video Description:
            {video_description}

            Format the output as valid JSON:
            {format_instructions}
            """
        )
        self.response_schema = ResponseSchema(
            name="questions",
            description="A list of exactly three questions related to the given question type."
        )
        self.parser = StructuredOutputParser.from_response_schemas([self.response_schema])

    def generate_questions(self, video_description: str) -> Dict[str, List[str]]:
        questions = {}
        for qtype in self.question_types:
            try:
                response = self.llm_model.run_chain(self.question_prompt, {
                    "question_type": qtype,
                    "video_description": video_description,
                    "format_instructions": self.parser.get_format_instructions()
                })
                parsed_output = self.parser.parse(response)
                questions[qtype] = parsed_output.get("questions", [])
            except Exception as e:
                questions[qtype] = []
        return questions
