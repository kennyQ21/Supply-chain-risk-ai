from chains.risk_analysis_chain import RiskAnalysisChain
# from chains.scenario_modeling_chain import ScenarioModelingChain
from chains.supplier_assessment_chain import SupplierAssessmentChain
from chains.alert_generation_chain import AlertGenerationChain
# from chains.financial_data_chain import FinancialDataChain
from chains.geospatial_risk_chain import GeospatialRiskChain

class ChainManager:
    """Central manager for all chains"""
    
    def __init__(self, llm, config):
        self.llm = llm
        self.config = config
        self.chains = self._initialize_chains()
    
    def _initialize_chains(self):
        return {
            "risk_analysis": RiskAnalysisChain(self.llm),
            "scenario_modeling": ScenarioModelingChain(self.llm),
            "supplier_assessment": SupplierAssessmentChain(self.llm),
            "alert_generation": AlertGenerationChain(self.llm),
            "financial_analysis": FinancialDataChain(self.llm),
            "geospatial_risk": GeospatialRiskChain(self.llm),
            "commodity_analysis": CommodityAnalysisChain(self.llm),
            "cyber_risk": CyberRiskChain(self.llm),
            "report_generation": ReportGenerationChain(self.llm),
            "dashboard_data": DashboardDataChain(self.llm)
        }
    
    def execute_analysis(self, data_input):
        """Execute full analysis pipeline"""
        results = {}
        
        # Execute chains in dependency order
        results["financial"] = self.chains["financial_analysis"].run(data_input)
        results["suppliers"] = self.chains["supplier_assessment"].run(data_input)
        results["scenarios"] = self.chains["scenario_modeling"].run(data_input)
        results["risks"] = self.chains["risk_analysis"].run(data_input)
        results["alerts"] = self.chains["alert_generation"].run(data_input)
        
        return results