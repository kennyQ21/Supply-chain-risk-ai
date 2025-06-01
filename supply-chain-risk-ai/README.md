# Supply Chain Risk Management AI System

An intelligent system for detecting, analyzing, and mitigating supply chain risks using LangChain, LangGraph, and LangSmith.

## 🚀 Quick Start

1. **Setup Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
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
