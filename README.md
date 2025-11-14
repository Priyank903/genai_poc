# Multi-Agent Post-Discharge Care Chatbot System

A sophisticated multi-agent AI system designed for post-discharge patient care in nephrology, featuring RAG (Retrieval-Augmented Generation) with comprehensive medical references and intelligent agent routing.

## 🏥 System Overview

This system implements a **two-agent architecture** with specialized workflows:

- **Receptionist Agent**: Handles patient identification, discharge information retrieval, and basic care guidance
- **Clinical AI Agent**: Provides medical expertise using RAG over nephrology references and web search fallback

## ✨ Key Features

### 🤖 Multi-Agent Architecture
- **Receptionist Agent**: Patient data retrieval, discharge info, routing decisions
- **Clinical AI Agent**: Medical Q&A, RAG-based responses, web search integration
- **Intelligent Routing**: Automatic handoff between agents based on query type

### 📚 RAG Implementation
- **43,473 medical text chunks** from comprehensive nephrology textbook
- **FAISS vector database** with cosine similarity search
- **Top-3 relevant chunk retrieval** with similarity thresholds
- **Source citations** and page references for all medical information

### 🔍 Dual Information Sources
- **Primary**: RAG over nephrology reference materials
- **Fallback**: Web search for queries outside reference scope
- **Clear source indication** (medical references vs. web search)

### 👥 Patient Data Management
- **26 dummy patient records** with realistic discharge information
- **Secure patient lookup** by name with fuzzy matching
- **Comprehensive discharge data**: medications, restrictions, follow-ups, warnings

### 🌐 Web Interface
- **Streamlit-based chat interface** with modern UI
- **Real-time conversation** with agent status indicators
- **Patient information display** with quick action buttons
- **Comprehensive logging** of all interactions

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit     │    │  Multi-Agent     │    │   Data Layer    │
│   Web Interface │◄──►│     System       │◄──►│                 │
│                 │    │                  │    │ • Patient DB    │
└─────────────────┘    │ ┌──────────────┐ │    │ • Vector DB     │
                       │ │ Receptionist │ │    │ • Logs          │
                       │ │    Agent     │ │    │                 │
                       │ └──────────────┘ │    └─────────────────┘
                       │ ┌──────────────┐ │    
                       │ │  Clinical    │ │    ┌─────────────────┐
                       │ │    Agent     │◄────►│  RAG System     │
                       │ └──────────────┘ │    │                 │
                       └──────────────────┘    │ • Embeddings    │
                                              │ • FAISS Index   │
                                              │ • Sarvam AI     │
                                              │ • Web Search    │
                                              └─────────────────┘
```

## 📁 Project Structure & File Descriptions

```
genai_poc/
├── data/                          # Data files
│   ├── patients.json              # 26 dummy patient discharge records with medical info
│   ├── nephro_chunks.json         # 43,473 processed nephrology text chunks from PDF
│   └── create_chunks_from_pdf.ipynb # Jupyter notebook for PDF text extraction
├── embeddings/                    # Vector database storage
│   ├── faiss_index/              # Legacy FAISS index (unused)
│   └── nephro_faiss/             # Active FAISS vector database
│       ├── nephro_index.faiss    # 66MB vector index with 43K embeddings
│       ├── nephro_metadata.pkl   # 10MB metadata (text, pages, sources)
│       └── index_info.json       # Index configuration and stats
├── logs/                          # System logs
│   ├── conversations.jsonl       # Structured conversation logs
│   └── interactions.log          # Detailed system interaction logs
├── src/                          # Source code
│   ├── improved_embeddings.py    # Creates FAISS embeddings from text chunks
│   ├── improved_rag_system.py    # RAG system with Sarvam AI integration
│   ├── multi_agent_system.py     # Core multi-agent orchestration & routing
│   ├── streamlit_chat_app.py     # Streamlit web interface
│   ├── fastapi_backend.py        # FastAPI REST API backend
│   └── web_search.py             # DuckDuckGo web search integration
├── run_api.py                    # FastAPI server startup script
├── .env                          # Environment variables (API keys)
└── requirements.txt              # Python dependencies
```

### 🔧 Key File Functions

**Core System Files:**
- **`multi_agent_system.py`** - Main orchestrator with ReceptionistAgent & ClinicalAgent
- **`improved_rag_system.py`** - Handles medical knowledge retrieval using FAISS + Sarvam AI
- **`streamlit_chat_app.py`** - User-friendly web chat interface
- **`fastapi_backend.py`** - REST API for programmatic access

**Data Processing:**
- **`improved_embeddings.py`** - Converts text chunks to vector embeddings
- **`web_search.py`** - Fallback web search when RAG context insufficient
- **`create_chunks_from_pdf.ipynb`** - Extracts and processes PDF content

**Data Storage:**
- **`patients.json`** - Patient discharge records with medications, restrictions
- **`nephro_chunks.json`** - Medical text chunks from nephrology textbook
- **`nephro_index.faiss`** - Vector embeddings for similarity search
- **`nephro_metadata.pkl`** - Text content and metadata for retrieved chunks

## 🚀 Quick Start

### 1. Installation

```bash
# Clone and navigate to project
cd genai_poc

# Install dependencies
pip install -r requirements.txt
```

### 2. Sarvam AI API Key Setup

**🆓 Free API Access**: Sarvam AI provides **1000 free API calls** for new users!

1. **Get your free API key** from [Sarvam AI](https://www.sarvam.ai/)
2. **Set as environment variable** (recommended for security):

**Windows (Git Bash/MINGW64):**
```bash
export SARVAM_API_KEY="your_sarvam_api_key_here"
```

**Windows (PowerShell):**
```powershell
$env:SARVAM_API_KEY="your_sarvam_api_key_here"
```

**Windows (System Environment Variable - Permanent):**
- Press `Win + R` → type `sysdm.cpl` → Advanced → Environment Variables
- Add: `SARVAM_API_KEY` = `your_sarvam_api_key_here`

**Linux/Mac:**
```bash
export SARVAM_API_KEY="your_sarvam_api_key_here"
```

3. **Verify setup:**
```bash
python -c "import os; print('API Key loaded:', 'Yes' if os.getenv('SARVAM_API_KEY') else 'No')"
```

### 3. Choose Your Interface

**Option A: Streamlit Web Interface (Recommended)**
```bash
streamlit run src/streamlit_chat_app.py
```

**Option B: FastAPI Backend**
```bash
python run_api.py
# Visit: http://localhost:8000/docs for API documentation
```

**Option C: Both (Full Stack)**
```bash
# Terminal 1: Start FastAPI backend
python run_api.py

# Terminal 2: Start Streamlit frontend  
streamlit run src/streamlit_chat_app.py
```

### 4. Build Embeddings (Optional - Already Included)

The system comes with pre-built embeddings, but you can rebuild them:
```bash
python src/improved_embeddings.py --chunks data/nephro_chunks.json --output embeddings/nephro_faiss
```

### 5. Test the System

```bash
# Test embeddings and RAG retrieval
python -c "from src.improved_rag_system import ImprovedRAGSystem; rag = ImprovedRAGSystem(); print('✅ RAG system loaded successfully')"

# Test multi-agent system
python -c "from src.multi_agent_system import MultiAgentSystem; mas = MultiAgentSystem(); print('✅ Multi-agent system loaded successfully')"
```

## 💬 Usage Examples

### Sample Conversation Flow

```
System: Hello! I'm your post-discharge care assistant...

User: My name is John Smith

Receptionist: Hi John Smith! I found your discharge report from 2024-01-15 
for Chronic Kidney Disease Stage 3. How are you feeling today?

User: I'm having swelling in my legs. Should I be worried?

System: This sounds like a medical concern. Let me connect you with our 
Clinical AI Agent.

Clinical Agent: Based on your CKD diagnosis and nephrology guidelines, 
leg swelling can indicate fluid retention...
[Provides detailed medical information with citations]
```

### Available Patient Names
- John Smith (CKD Stage 3)
- Sarah Johnson (Acute Kidney Injury)
- Michael Chen (Nephrolithiasis)
- Emily Rodriguez (Diabetic Nephropathy)
- [22 more patients with various nephrology conditions]

## 🔧 Technical Specifications

### RAG System
- **Model**: SentenceTransformers (all-MiniLM-L6-v2)
- **Vector DB**: FAISS with cosine similarity
- **Chunks**: 43,473 medical text segments
- **Retrieval**: Top-3 with similarity threshold ≥ 0.3
- **LLM**: Sarvam AI with medical prompt engineering

### Multi-Agent Framework
- **Custom Implementation** with LangChain components
- **State Management**: Conversation context preservation
- **Agent Routing**: Keyword-based medical query detection
- **Logging**: Comprehensive interaction tracking

### Data Storage
- **Patient Data**: JSON files with structured discharge information
- **Vector Storage**: FAISS index with metadata
- **Logs**: JSON-based interaction history
- **Embeddings**: Pickle serialization for metadata

## 📊 System Performance

### RAG Retrieval Results
```
Query: "chronic kidney disease symptoms"
├── Found: 3 relevant chunks
├── Similarity scores: 0.860, 0.860, 0.841
└── Sources: Pages 96, 96, 1154

Query: "nephron anatomy and function"  
├── Found: 3 relevant chunks
├── Similarity scores: 0.765, 0.738, 0.721
└── Sources: Pages 19, 950, 2
```

### Agent Performance
- **Patient Identification**: 100% accuracy for exact name matches
- **Medical Query Routing**: Keyword-based detection with high precision
- **Response Generation**: RAG + Sarvam AI with web search fallback
- **Logging Coverage**: All interactions tracked with timestamps

## 🔒 Medical Disclaimers

**⚠️ IMPORTANT**: This system is for **educational purposes only**. All responses include appropriate medical disclaimers:

- Not a replacement for professional medical advice
- Always consult healthcare professionals for medical guidance
- Information provided for educational purposes only
- Emergency situations require immediate medical attention

## 🛠️ Development & Testing

### Running Tests

```bash
# Test embeddings and RAG retrieval
python -c "from src.improved_rag_system import ImprovedRAGSystem; rag = ImprovedRAGSystem(); print('✅ RAG system loaded successfully')"

# Test multi-agent system
python -c "from src.multi_agent_system import MultiAgentSystem; mas = MultiAgentSystem(); print('✅ Multi-agent system loaded successfully')"

# Test individual RAG queries
python -c "
from src.improved_rag_system import ImprovedRAGSystem
rag = ImprovedRAGSystem()
result = rag.retrieve_relevant_chunks('What are nephrons?', top_k=3)
print(f'Found {len(result)} relevant chunks')
for i, chunk in enumerate(result, 1):
    print(f'{i}. Score: {chunk[\"similarity_score\"]:.3f}, Page: {chunk[\"page\"]}')
"
```

### Troubleshooting

**Common Issues:**

1. **"API Key loaded: No"**
   ```bash
   # Set environment variable in your current session
   export SARVAM_API_KEY="your_key_here"  # Git Bash/Linux/Mac
   $env:SARVAM_API_KEY="your_key_here"    # PowerShell
   ```

2. **"FAISS index not found"**
   ```bash
   # Rebuild embeddings
   python src/improved_embeddings.py --chunks data/nephro_chunks.json --output embeddings/nephro_faiss
   ```

3. **Import errors**
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt
   ```

4. **Streamlit not starting**
   ```bash
   # Check if port is available
   streamlit run src/streamlit_chat_app.py --server.port 8502
   ```

### Adding New Patients

Add to `data/patients.json`:
```json
{
  "patient_name": "New Patient",
  "discharge_date": "2024-02-15",
  "primary_diagnosis": "Condition Name",
  "medications": ["Med1", "Med2"],
  "dietary_restrictions": "Restrictions",
  "follow_up": "Follow-up schedule",
  "warning_signs": "Signs to watch",
  "discharge_instructions": "Instructions"
}
```

### Extending Medical Knowledge

1. Add new medical texts to `data/nephro_chunks.json`
2. Rebuild embeddings: `python src/improved_embeddings.py`
3. Test retrieval: `python test_rag.py`

## 📈 Future Enhancements

- **Voice Interface**: Speech-to-text integration
- **Mobile App**: React Native or Flutter implementation
- **Advanced NLP**: Fine-tuned medical language models
- **Integration**: EHR system connectivity
- **Analytics**: Patient interaction insights
- **Multilingual**: Support for multiple languages

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit pull request

## 📄 License

This project is for educational purposes. Please ensure compliance with healthcare regulations (HIPAA, etc.) before production use.

## 🆘 Support & FAQ

### Frequently Asked Questions

**Q: Do I need to pay for Sarvam AI?**
A: No! Sarvam AI provides **1000 free API calls** for new users. Perfect for testing and development.

**Q: Can I run this without an API key?**
A: Yes! The system will fall back to local models and web search, but responses may be less sophisticated.

**Q: How do I add more medical content?**
A: Add text to `data/nephro_chunks.json` and rebuild embeddings with `python src/improved_embeddings.py`

**Q: Can I use this for real patients?**
A: This is for **educational purposes only**. For production use, ensure HIPAA compliance and proper medical validation.

### Getting Help

For issues or questions:
1. **Check logs**: `logs/conversations.jsonl` and `logs/interactions.log`
2. **Test components**: Use the test commands in the Development section
3. **Verify setup**: Ensure API key is set and dependencies installed
4. **Check ports**: Make sure ports 8501 (Streamlit) and 8000 (FastAPI) are available

---

**Built with**: Python, Streamlit, FAISS, SentenceTransformers, Sarvam AI
**Purpose**: Educational demonstration of multi-agent RAG systems in healthcare
