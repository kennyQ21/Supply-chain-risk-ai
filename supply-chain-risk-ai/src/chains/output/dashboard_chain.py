from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class DashboardDataChain:
    """Process data for dashboard visualization"""
    
    def __init__(self, llm):
        self.kpi_chain = self._create_kpi_chain()
        self.visualization_chain = self._create_visualization_chain()
    
    def _create_kpi_chain(self):
        prompt = PromptTemplate(
            input_variables=["raw_data", "historical_benchmarks"],
            template="""
            Extract key performance indicators for dashboard:
            
            Raw Data: {raw_data}
            Historical Benchmarks: {historical_benchmarks}
            
            Generate KPIs:
            1. Overall Risk Score (1-10)
            2. Risk Trend (improving/stable/deteriorating)
            3. Supplier Health Score
            4. Geographic Risk Distribution
            5. Alert Priority Breakdown
            6. Cost Impact Estimates
            
            Format for real-time dashboard display.
            """
        )
        return LLMChain(llm=self.llm, prompt=prompt)