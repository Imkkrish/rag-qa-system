import google.generativeai as genai
from core_logic.core.config import settings
from typing import List

class LLMService:
    def __init__(self):
        if settings.GOOGLE_API_KEY:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None

    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        if not self.model:
            return "LLM service not configured. Please provide GOOGLE_API_KEY."
            
        context_text = "\n\n".join([chunk['text'] for chunk in context_chunks])
        
        prompt = f"""
        You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.

        Context:
        {context_text}

        Question: {query}

        Answer:
        """
        
        response = self.model.generate_content(prompt)
        return response.text

llm_service = LLMService()
