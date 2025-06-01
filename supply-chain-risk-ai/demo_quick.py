"""
Quick Demo - Fast Supply Chain Risk Test
Run with: python demo_quick.py
"""

try:
    from langchain_ollama import OllamaLLM
except ImportError:
    from langchain_community.llms import Ollama as OllamaLLM

from langchain_core.prompts import PromptTemplate
from faker import Faker
import pandas as pd
import numpy as np

def quick_test():
    """Quick test with minimal data"""
    print("🚀 Quick Supply Chain Risk Test...")
    
    # Generate minimal data
    fake = Faker()
    suppliers = []
    for i in range(3):
        suppliers.append({
            'name': fake.company(),
            'risk_score': np.random.randint(20, 90),
        })
    
    df = pd.DataFrame(suppliers)
    print("Sample Suppliers:")
    print(df)
    print()
    
    # Setup fast LLM
    llm = OllamaLLM(
        model="llama3.2", 
        base_url="http://localhost:11434",
        temperature=0,
        num_predict=50  # Short response
    )
    
    # Simple prompt
    prompt = PromptTemplate(
        input_variables=["data"],
        template="Which supplier has highest risk? {data}"
    )
    
    chain = prompt | llm
    
    print("🤖 Quick AI analysis...")
    result = chain.invoke({"data": df.to_string()})
    print(f"Result: {result}")

if __name__ == "__main__":
    quick_test()
