#!/usr/bin/env python3
"""
Supply Chain Risk Management System - Project Setup Script
Run this script to set up your project structure and initial configuration
"""

import os
import json
from pathlib import Path

def create_project_structure():
    """Create the complete project directory structure"""
    
    # Define project structure
    directories = [
        "data/raw",
        "data/processed", 
        "data/synthetic",
        "data/external",
        "src/chains",
        "src/graphs", 
        "src/utils",
        "src/api",
        "src/dashboard",
        "docs",
        "tests",
        "notebooks",
        "config",
        "logs"
    ]
    
    # Create directories
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def create_config_files():
    """Create initial configuration files"""
    
    # Requirements file
    requirements = """
# Core LangChain ecosystem
langchain==0.1.0
langgraph==0.0.40
langsmith==0.0.83

# Local LLM and embeddings
ollama
sentence-transformers
chromadb

# Web framework and API
streamlit
fastapi
uvicorn

# Data processing
pandas
numpy
matplotlib
seaborn
plotly

# External data sources
requests
beautifulsoup4
feedparser
python-dotenv

# Synthetic data generation
faker

# Database
sqlite3
sqlalchemy

# Utilities
python-dateutil
pydantic
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements.strip())
    print("✓ Created requirements.txt")
    
    # Environment variables template
    env_template = """
# API Keys (all optional for free tier)
OPENWEATHER_API_KEY=your_free_key_here
NEWS_API_KEY=your_free_key_here
ALPHA_VANTAGE_KEY=demo

# LangSmith (free tier)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_free_langsmith_key

# Local LLM settings
OLLAMA_MODEL=llama3.2
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Database
DATABASE_URL=sqlite:///supply_chain.db

# Application settings
DEBUG=true
LOG_LEVEL=INFO
"""
    
    with open(".env.example", "w") as f:
        f.write(env_template.strip())
    print("✓ Created .env.example")
    
    # Main configuration
    config = {
        "project": {
            "name": "Supply Chain Risk Management AI",
            "version": "0.1.0",
            "description": "Intelligent supply chain risk detection and mitigation system"
        },
        "llm": {
            "provider": "ollama",
            "model": "llama3.1:1b",
            "temperature": 0.1,
            "max_tokens": 2000
        },
        "embeddings": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "chunk_size": 500,
            "chunk_overlap": 50
        },
        "risk_thresholds": {
            "low": 30,
            "medium": 60, 
            "high": 80,
            "critical": 95
        },
        "data_sources": {
            "news_feeds": [
                "https://feeds.reuters.com/reuters/businessNews",
                "https://rss.cnn.com/rss/money_news_international.rss"
            ],
            "weather_api": "openweathermap",
            "financial_api": "alpha_vantage"
        }
    }
    
    with open("config/config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("✓ Created config/config.json")

def create_initial_files():
    """Create initial Python files with basic structure"""
    
    # Main application entry point
    main_app = '''"""
Supply Chain Risk Management System
Main application entry point
"""

import streamlit as st
from src.dashboard.main_dashboard import run_dashboard

if __name__ == "__main__":
    st.set_page_config(
        page_title="Supply Chain Risk AI",
        page_icon="🚚",
        layout="wide"
    )
    
    run_dashboard()
'''
    
    with open("app.py", "w") as f:
        f.write(main_app)
    print("✓ Created app.py")
    
    # Utility functions
    utils_init = '''"""
Utility functions for the supply chain risk management system
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_config():
    """Load configuration from config.json"""
    config_path = Path("config/config.json")
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}

def get_env_var(key, default=None):
    """Get environment variable with default"""
    return os.getenv(key, default)

def setup_logging():
    """Setup logging configuration"""
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/app.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# Global configuration
CONFIG = load_config()
logger = setup_logging()
'''
    
    with open("src/utils/__init__.py", "w") as f:
        f.write(utils_init)
    print("✓ Created src/utils/__init__.py")
    
    # Create __init__.py files
    init_files = [
        "src/__init__.py",
        "src/chains/__init__.py", 
        "src/graphs/__init__.py",
        "src/api/__init__.py",
        "src/dashboard/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"✓ Created {init_file}")

def create_readme():
    """Create project README"""
    
    readme = '''# Supply Chain Risk Management AI System

An intelligent system for detecting, analyzing, and mitigating supply chain risks using LangChain, LangGraph, and LangSmith.

## 🚀 Quick Start

1. **Setup Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\\Scripts\\activate on Windows
   pip install -r requirements.txt
   ```

2. **Install Local LLM**
   ```bash
   # Install Ollama from ollama.ai
   ollama pull llama3.2
   ```

3. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys (optional for basic functionality)
   ```

4. **Run Application**
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure

```
supply-chain-risk-ai/
├── src/
│   ├── chains/          # LangChain components
│   ├── graphs/          # LangGraph workflows  
│   ├── utils/           # Utility functions
│   ├── api/             # FastAPI endpoints
│   └── dashboard/       # Streamlit dashboard
├── data/
│   ├── raw/             # Original data files
│   ├── processed/       # Cleaned data
│   ├── synthetic/       # Generated data
│   └── external/        # API data cache
├── config/              # Configuration files
├── notebooks/           # Jupyter notebooks
├── tests/               # Unit tests
└── docs/                # Documentation
```

## 🎯 Features

- **Risk Detection**: Real-time monitoring of supply chain risks
- **Impact Assessment**: Automated analysis of potential disruptions  
- **Mitigation Planning**: AI-powered recommendation engine
- **Human Oversight**: Approval workflows for critical decisions
- **Performance Monitoring**: Comprehensive system observability

## 🛠️ Built With

- **LangChain**: Document processing and LLM chains
- **LangGraph**: Stateful workflow orchestration
- **LangSmith**: System monitoring and evaluation
- **Ollama**: Local LLM inference
- **Streamlit**: Web dashboard
- **ChromaDB**: Vector database for embeddings

## 📈 Learning Journey

This project demonstrates:
- End-to-end AI system development
- Supply chain domain expertise
- Production-ready architecture
- Cost-effective implementation using open source tools

## 🤝 Contributing

This is a learning project. Feel free to extend and modify for your own educational purposes.

## 📄 License

MIT License - see LICENSE file for details.
'''
    
    with open("README.md", "w") as f:
        f.write(readme)
    print("✓ Created README.md")

def main():
    """Run the complete project setup"""
    print("🚀 Setting up Supply Chain Risk Management AI System...")
    print()
    
    create_project_structure()
    print()
    
    create_config_files()
    print()
    
    create_initial_files()
    print()
    
    create_readme()
    print()
    
    print("✅ Project setup complete!")
    print()
    print("Next steps:")
    print("1. Install requirements: pip install -r requirements.txt")
    print("2. Install Ollama and pull model: ollama pull llama3.2")
    print("3. Copy .env.example to .env and configure")
    print("4. Run the setup verification script")

if __name__ == "__main__":
    main()