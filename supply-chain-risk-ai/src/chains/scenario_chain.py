from langchain.prompts import PromptTemplate

class ScenarioModelingChain:
    """Chain for running what-if scenario analysis"""
    
    def __init__(self, llm):
        self.llm = llm
        self.scenario_prompt = PromptTemplate(
            input_variables=["base_data", "scenario_type", "parameters"],
            template="""
            Given the current supply chain state: {base_data}
            
            Model the following scenario: {scenario_type}
            Parameters: {parameters}
            
            Calculate:
            1. Probability of occurrence
            2. Immediate impact (financial, operational)
            3. Cascading effects
            4. Recovery timeline
            5. Mitigation strategies
            """
        )