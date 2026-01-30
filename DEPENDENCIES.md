# Dependencies Guide

This document provides guidance on installing dependencies for the RAG system's three main knowledge bases.

## Dependency Files

| File Name | Purpose | Target Knowledge Base |
|-----------|---------|------------------------|
| `requirements.txt` | Complete set of dependencies for the entire project | All knowledge bases |
| `requirements_kb.txt` | Dependencies for knowledge base construction | Core knowledge base |
| `requirements_retrieval.txt` | Dependencies for retrieval system | Retrieval knowledge base |
| `requirements_generation.txt` | Dependencies for generation and evaluation | Generation/evaluation knowledge base |

## Installation Options

### Option 1: Full Installation (All Knowledge Bases)

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Install SpaCy model
python -m spacy download en_core_web_sm

# Install Ollama (required for answer generation)
# Visit https://ollama.ai/download and follow installation instructions

# Pull required model
ollama pull qwen2:7b-instruct
```

### Option 2: Minimal Installation (Core Functionality)

```bash
# Install only core dependencies
pip install -r requirements_kb.txt
pip install -r requirements_retrieval.txt

# Install SpaCy model
python -m spacy download en_core_web_sm
```

### Option 3: Development Installation

```bash
# Install full dependencies plus development tools
pip install -r requirements.txt
pip install pytest black flake8 isort
```

## Knowledge Base Specific Installation

### 1. Core Knowledge Base (KB Builder)

**Purpose**: Constructing the knowledge base from raw data

**Dependencies**: `requirements_kb.txt`

**Installation**:
```bash
pip install -r requirements_kb.txt
```

**Key Components**:
- Data processing (pandas, numpy)
- Optional embedding models (modelscope)
- Optional vector indexing (faiss-cpu)

### 2. Retrieval Knowledge Base

**Purpose**: Performing efficient retrieval using various methods

**Dependencies**: `requirements_retrieval.txt`

**Installation**:
```bash
pip install -r requirements_retrieval.txt
python -m spacy download en_core_web_sm
```

**Key Components**:
- Vector retrieval (faiss-cpu)
- Keyword retrieval (rank_bm25)
- Entity extraction (spacy)
- Optional graph retrieval (networkx)

### 3. Generation/Evaluation Knowledge Base

**Purpose**: Generating answers and evaluating system performance

**Dependencies**: `requirements_generation.txt`

**Installation**:
```bash
pip install -r requirements_generation.txt
```

**Key Components**:
- Evaluation metrics (scikit-learn, nltk, rouge)
- Optional visualization (matplotlib, seaborn)
- Ollama (required for answer generation, installed separately)

## Ollama Setup

Ollama is required for answer generation but must be installed separately:

1. **Install Ollama**:
   - Visit https://ollama.ai/download
   - Follow the installation instructions for your operating system

2. **Start Ollama Service**:
   ```bash
   # On Linux/macOS
   ollama serve
   
   # On Windows, the service starts automatically
   ```

3. **Pull Required Model**:
   ```bash
   ollama pull qwen2:7b-instruct
   ```

4. **Verify Installation**:
   ```bash
   ollama list
   # Should show qwen2:7b-instruct
   ```

## FAISS IVF Index Setup

For optimal retrieval performance with large knowledge bases:

1. **Install FAISS**:
   ```bash
   pip install faiss-cpu>=1.7.4
   ```

2. **Configure IVF Index**:
   - The knowledge base builder will automatically create IVF indexes
   - Default configuration: `nlist=100` for IVF index

## BM25 Setup

For keyword-based retrieval:

1. **Install Rank-BM25**:
   ```bash
   pip install rank_bm25
   ```

2. **BM25 will be automatically initialized during knowledge base construction**

## Troubleshooting

### Common Issues

1. **Ollama connection failed**:
   - Ensure Ollama service is running: `ollama serve`
   - Check model is available: `ollama list`

2. **FAISS import error**:
   - Use faiss-cpu instead of faiss-gpu if no GPU is available
   - Try `pip install --no-cache-dir faiss-cpu`

3. **SpaCy model not found**:
   - Run: `python -m spacy download en_core_web_sm`
   - Check installation: `python -c "import spacy; print(spacy.load('en_core_web_sm'))"`

4. **ModelScope import error**:
   - This is optional, the system will work without it
   - Try installing specific version: `pip install modelscope==1.9.0`

### Dependency Version Conflicts

If you encounter version conflicts:

1. **Create a fresh virtual environment**:
   ```bash
   rm -rf venv
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies in order**:
   ```bash
   pip install pandas numpy requests
   pip install -r requirements.txt
   ```

3. **Check installed versions**:
   ```bash
   pip list | grep -E 'pandas|numpy|faiss|spacy'
   ```

## Recommended Versions

| Package | Minimum Version | Recommended Version |
|---------|----------------|----------------------|
| Python | 3.8 | 3.10+ |
| pandas | 2.0.0 | 2.1.0+ |
| numpy | 1.24.0 | 1.26.0+ |
| faiss-cpu | 1.7.4 | 1.7.4 |
| spacy | 3.6.0 | 3.7.0+ |
| ollama | N/A | Latest |
| qwen2:7b-instruct | N/A | Latest |

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8GB | 16GB+ |
| CPU | 4 cores | 8 cores+ |
| Storage | 10GB | 50GB+ (for large knowledge bases) |
| GPU | None (CPU-only) | Optional (for faster embedding generation) |

## Network Requirements

- **Internet connection** required for:
  - Installing dependencies
  - Downloading SpaCy models
  - Pulling Ollama models
  - Optional: Using online embedding models

- **Local network** required for:
  - Ollama service communication (http://localhost:11434)

## Offline Installation

To install dependencies offline:

1. **Download dependencies on a networked machine**:
   ```bash
   pip download -r requirements.txt -d dependencies/
   ```

2. **Transfer the `dependencies/` folder to the offline machine**

3. **Install from local files**:
   ```bash
   pip install --no-index --find-links=dependencies/ -r requirements.txt
   ```

4. **Manually install Ollama and SpaCy models** (requires separate download)

## Conclusion

The RAG system's dependencies are structured to support its three main knowledge bases:
- **Core knowledge base** for information storage
- **Retrieval knowledge base** for efficient information access
- **Generation/evaluation knowledge base** for producing answers and assessing performance

By following the installation instructions above, you can set up the appropriate dependencies for your specific use case.
