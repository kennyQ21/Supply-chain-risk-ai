from langchain.prompts import PromptTemplate

class AlertGenerationChain:
    """Chain for generating risk alerts and recommendations"""
    
    def __init__(self, llm):
        self.llm = llm
        self.alert_prompt = PromptTemplate(
            input_variables=["risk_scores", "thresholds", "historical_data"],
            template="""
            Generate alerts based on:
            Current Risk Scores: {risk_scores}
            Alert Thresholds: {thresholds}
            Historical Context: {historical_data}
            
            For each alert, provide:
            1. Severity level (Low/Medium/High/Critical)
            2. Specific trigger condition
            3. Immediate action required
            4. Timeline for resolution
            5. Escalation path
            """
        )