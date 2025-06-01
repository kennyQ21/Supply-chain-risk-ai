from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class GeospatialRiskChain:
    """Analyze location-based risks"""
    
    def __init__(self, llm):
        self.llm = llm
        self.location_risk_chain = self._create_location_risk_chain()
        self.disaster_impact_chain = self._create_disaster_impact_chain()
    
    def _create_location_risk_chain(self):
        prompt = PromptTemplate(
            input_variables=["coordinates", "supplier_data", "historical_events"],
            template="""
            Assess geographic risk for location: {coordinates}
            Supplier Data: {supplier_data}
            Historical Events: {historical_events}
            
            Evaluate:
            1. Natural disaster probability
            2. Political stability
            3. Infrastructure reliability
            4. Climate change impacts
            5. Transportation vulnerabilities
            """
        )
        return LLMChain(llm=self.llm, prompt=prompt)