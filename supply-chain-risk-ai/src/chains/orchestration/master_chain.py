from langchain.chains import SequentialChain
from chains.data.financial_chain import FinancialDataChain  # Make sure this import path is correct
from chains.supplier_chain import SupplierAssessmentChain  # Make sure this import path is correct
from chains.scenario_chain import ScenarioModelingChain  # Make sure this import path is correct
from chains.alert_chain import AlertGenerationChain  # Make sure this import path is correct
from chains.risk_analysis_chain import RiskAnalysisChain  # Updated import path to match actual module name

class MasterAnalysisChain:
    """Orchestrates all analysis chains"""
    
    def __init__(self, llm):
        self.llm = llm
        self.financial_chain = FinancialDataChain(llm)
        self.supplier_chain = SupplierAssessmentChain(llm)
        self.scenario_chain = ScenarioModelingChain(llm)
        self.risk_chain = RiskAnalysisChain(llm)
        self.alert_chain = AlertGenerationChain(llm)
        
    def create_master_chain(self):
        return SequentialChain(
            chains=[
                self.financial_chain,
                self.supplier_chain,
                self.scenario_chain,
                self.risk_chain,
                self.alert_chain
            ],
            input_variables=[
                "financial_data", 
                "supplier_data", 
                "external_risks",
                "geospatial_data"
            ],
            output_variables=[
                "risk_assessment",
                "scenarios",
                "alerts",
                "recommendations"
            ]
        )