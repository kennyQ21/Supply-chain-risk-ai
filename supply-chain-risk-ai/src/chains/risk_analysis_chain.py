from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from typing import Dict, List, Any

class RiskAnalysisChain:
    """Main chain for analyzing supply chain risks"""
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["financial_data", "supplier_data", "external_risks"],
            template="""
            Analyze the following supply chain risk data:
            
            Financial Metrics: {financial_data}
            Supplier Network: {supplier_data}
            External Risk Factors: {external_risks}
            
            Provide a comprehensive risk assessment including:
            1. Overall risk score (1-10)
            2. Key risk factors
            3. Immediate recommendations
            4. Long-term strategic suggestions
            """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)