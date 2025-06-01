from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class ReportGenerationChain:
    """Generate comprehensive risk reports"""
    
    def __init__(self, llm):
        self.executive_summary_chain = self._create_executive_summary_chain()
        self.detailed_analysis_chain = self._create_detailed_analysis_chain()
        self.recommendations_chain = self._create_recommendations_chain()
    
    def _create_executive_summary_chain(self):
        prompt = PromptTemplate(
            input_variables=["risk_analysis", "key_metrics", "critical_alerts"],
            template="""
            Create executive summary for supply chain risk report:
            
            Risk Analysis: {risk_analysis}
            Key Metrics: {key_metrics}
            Critical Alerts: {critical_alerts}
            
            Summary should include:
            1. Overall risk assessment (2-3 sentences)
            2. Top 3 immediate concerns
            3. Key performance indicators
            4. Strategic recommendations
            5. Next review timeline
            
            Keep concise and actionable for C-level executives.
            """
        )
        return LLMChain(llm=self.llm, prompt=prompt)