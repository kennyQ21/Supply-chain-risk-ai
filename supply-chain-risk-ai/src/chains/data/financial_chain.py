from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class FinancialDataChain:
    """Process and analyze financial metrics"""
    
    def __init__(self, llm):
        self.price_analysis_chain = self._create_price_analysis_chain()
        self.volatility_chain = self._create_volatility_chain()
        self.correlation_chain = self._create_correlation_chain()
    
    def _create_price_analysis_chain(self):
        prompt = PromptTemplate(
            input_variables=["price_data", "volume_data", "market_context"],
            template="""
            Analyze price movements:
            Price Data: {price_data}
            Volume Data: {volume_data}
            Market Context: {market_context}
            
            Determine:
            1. Trend direction and strength
            2. Support/resistance levels
            3. Volume confirmation
            4. Risk implications for supply chain
            """
        )
        return LLMChain(llm=self.llm, prompt=prompt)