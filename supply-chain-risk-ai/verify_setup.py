#!/usr/bin/env python3
"""
Faster Setup Verification Script
Optimized for speed with shorter tests
"""

import sys
import subprocess
import importlib
import requests
import time

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Need 3.8+")
        return False

def check_packages():
    """Check if required packages are installed"""
    print("\n📦 Checking required packages...")
    
    required_packages = [
        'langchain',
        'streamlit',
        'pandas',
        'numpy',
        'requests',
        'faker',
        'chromadb',
        'sentence_transformers'
    ]
    
    success = True
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - missing")
            success = False
    
    return success

def check_ollama():
    """Check if Ollama is installed and running"""
    print("\n🤖 Checking Ollama...")
    
    try:
        # Quick check if Ollama service is running
        response = requests.get('http://localhost:11434/api/tags', timeout=3)
        if response.status_code == 200:
            print("✅ Ollama service - running")
            
            # Check if model exists
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            
            if any('llama3.2' in model for model in models):
                print("✅ llama3.2 model - available")
                return True
            else:
                print("⚠️ llama3.2 model - not found")
                print("💡 Run: ollama pull llama3.2")
                return False
        else:
            print("❌ Ollama service - not responding")
            print("💡 Start with: ollama serve")
            return False
            
    except requests.exceptions.RequestException:
        print("❌ Ollama service - not running")
        print("💡 Start with: ollama serve")
        return False

def test_langchain_basic():
    """Test basic LangChain functionality - SETUP ONLY"""
    print("\n🔗 Testing LangChain setup...")
    
    try:
        # Try new import first
        try:
            from langchain_ollama import OllamaLLM
            print("✅ langchain_ollama imported")
        except ImportError:
            from langchain_community.llms import Ollama as OllamaLLM
            print("⚠️ using langchain_community (deprecated)")
        
        from langchain_core.prompts import PromptTemplate
        
        # Test if we can create instances (no actual LLM calls)
        llm = OllamaLLM(model="llama3.2", base_url="http://localhost:11434")
        prompt = PromptTemplate(
            input_variables=["topic"],
            template="What is {topic}?"
        )
        
        chain = prompt | llm
        print("✅ LangChain setup - OK")
        
        return True
        
    except Exception as e:
        print(f"❌ LangChain setup - Error: {str(e)}")
        return False

def test_quick_llm():
    """Quick LLM test with very short prompt"""
    print("\n🧪 Quick LLM test...")
    
    try:
        try:
            from langchain_ollama import OllamaLLM
        except ImportError:
            from langchain_community.llms import Ollama as OllamaLLM
        
        # Use a very simple, short prompt
        llm = OllamaLLM(
            model="llama3.2", 
            base_url="http://localhost:11434",
            temperature=0,  # Faster inference
            num_predict=10  # Limit response length
        )
        
        print("🔄 Testing with short prompt...")
        start_time = time.time()
        
        # Very simple prompt
        response = llm.invoke("Hi")
        
        end_time = time.time()
        print(f"✅ LLM responds in {end_time - start_time:.1f}s: '{response.strip()[:50]}...'")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM test failed: {str(e)}")
        return False

def test_data_generation():
    """Test synthetic data generation"""
    print("\n📊 Testing data generation...")
    
    try:
        from faker import Faker
        import pandas as pd
        import numpy as np
        
        fake = Faker()
        
        # Generate just 3 suppliers for speed
        suppliers = []
        for i in range(3):
            suppliers.append({
                'id': i,
                'name': fake.company(),
                'location': fake.city(),
                'risk_score': np.random.randint(1, 100)
            })
        
        df = pd.DataFrame(suppliers)
        print(f"✅ Generated {len(df)} sample suppliers")
        
        return True
        
    except Exception as e:
        print(f"❌ Data generation failed: {str(e)}")
        return False

def create_quick_demo():
    """Create a quick demo that runs fast"""
    print("\n🎯 Creating quick demo...")
    
    demo_code = '''"""
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
'''
    
    with open("demo_quick.py", "w") as f:
        f.write(demo_code)
    
    print("✅ Created demo_quick.py (fast demo)")
    print("💡 Run with: python demo_quick.py")

def warm_up_ollama():
    """Warm up Ollama model to make subsequent calls faster"""
    print("\n🔥 Warming up Ollama model...")
    
    try:
        # Send a simple request to warm up the model
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'llama3.2',
                'prompt': 'Hi',
                'stream': False,
                'options': {
                    'num_predict': 1,
                    'temperature': 0
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ Model warmed up - future calls will be faster")
            return True
        else:
            print("⚠️ Model warm-up failed")
            return False
            
    except Exception as e:
        print(f"⚠️ Warm-up failed: {e}")
        return False

def main():
    """Run fast verification checks"""
    print("🔍 Fast Supply Chain Risk AI Setup Check...")
    print("="*50)
    
    # Install langchain-ollama if missing
    try:
        importlib.import_module('langchain_ollama')
    except ImportError:
        print("📦 Installing langchain-ollama...")
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-U', 'langchain-ollama'
        ], capture_output=True)
    
    # Run basic checks (fast)
    checks = [
        ("Python", check_python_version()),
        ("Packages", check_packages()),
        ("Ollama", check_ollama()),
        ("LangChain", test_langchain_basic()),
        ("Data Gen", test_data_generation())
    ]
    
    print("\n" + "="*30)
    
    passed = sum(result for _, result in checks)
    total = len(checks)
    
    for name, result in checks:
        status = "✅" if result else "❌"
        print(f"{name:<10} {status}")
    
    print(f"\nBasic setup: {passed}/{total} ✅")
    
    if passed == total:
        print("\n🎉 Basic setup complete!")
        
        # Ask user if they want to test LLM (slow)
        print("\n🤔 Run LLM test? (takes 10-30 seconds)")
        choice = input("Type 'y' to test, or press Enter to skip: ").lower()
        
        if choice == 'y':
            warm_up_ollama()
            test_quick_llm()
        else:
            print("⏭️ LLM test skipped")
        
        create_quick_demo()
        print("\n🚀 Ready to go!")
        print("• Quick test: python demo_quick.py")
        
    else:
        print("⚠️ Fix issues above first")
        if not checks[2][1]:  # Ollama failed
            print("\n💡 Start Ollama: ollama serve")

if __name__ == "__main__":
    main()