from langchain.prompts import PromptTemplate

class CommodityAnalysisChain:
    """Analyze commodity market risks"""
    
    def __init__(self, llm):
        self.llm = llm
        self.commodity_prompt = PromptTemplate(
            input_variables=["commodity_data", "supply_demand", "geopolitical_events"],
            template="""
            Analyze commodity market for supply chain impact:
            
            Commodity Data: {commodity_data}
            Supply/Demand Factors: {supply_demand}
            Geopolitical Events: {geopolitical_events}
            
            Assess:
            1. Price volatility trends
            2. Supply disruption risks
            3. Substitution possibilities
            4. Long-term availability
            5. Strategic sourcing recommendations
            """
        )