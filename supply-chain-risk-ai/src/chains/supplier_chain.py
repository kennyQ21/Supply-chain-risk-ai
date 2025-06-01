from langchain.prompts import PromptTemplate

class SupplierAssessmentChain:
    """Chain for evaluating individual suppliers"""
    
    def __init__(self, llm):
        self.llm = llm
        self.supplier_prompt = PromptTemplate(
            input_variables=["supplier_data", "market_conditions", "geopolitical_factors"],
            template="""
            Assess supplier risk profile:
            
            Supplier Information: {supplier_data}
            Market Conditions: {market_conditions}
            Geopolitical Factors: {geopolitical_factors}
            
            Evaluate:
            1. Financial stability
            2. Operational reliability
            3. Geographic risk exposure
            4. Alternative supplier options
            5. Recommended monitoring frequency
            """
        )