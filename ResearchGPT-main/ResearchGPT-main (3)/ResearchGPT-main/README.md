# ResearchGPT Assistant

## Overview

ResearchGPT Assistant is an intelligent research tool that leverages advanced AI techniques to help researchers process academic documents, generate insights, and automate research workflows. This project demonstrates the integration of machine learning fundamentals, natural language processing, advanced prompting strategies, and AI agents in a practical research assistance application.

## Features

### Core Capabilities
- **Document Processing**: Extract and process text from PDF research papers
- **Intelligent Search**: TF-IDF based similarity search for relevant document retrieval
- **Advanced Prompting**: Chain-of-Thought, Self-Consistency, and ReAct prompting strategies
- **AI Agents**: Specialized agents for summarization, question-answering, and research workflows
- **Research Automation**: Complete research session management with multi-step workflows

### Advanced Prompting Techniques
- **Chain-of-Thought Reasoning**: Step-by-step logical reasoning for complex questions
- **Self-Consistency**: Multiple reasoning paths with consensus-based answers
- **ReAct Workflows**: Structured research processes with Thought-Action-Observation cycles
- **Verification and Editing**: Answer quality checking and improvement mechanisms

### AI Agents
- **Summarizer Agent**: Document and literature overview generation
- **QA Agent**: Factual and analytical question answering
- **Research Workflow Agent**: Complete research session orchestration
- **Agent Orchestrator**: Multi-agent task coordination and routing

## Technical Architecture

### Technology Stack
- **Python 3.8+**: Core programming language
- **Mistral API**: Large Language Model integration
- **scikit-learn**: Machine learning algorithms (TF-IDF, similarity search)
- **PyPDF2**: PDF text extraction
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing

### Project Structure
```
research_gpt_assistant/
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── config.py                      # Configuration and settings
├── document_processor.py          # PDF processing and text extraction
├── research_assistant.py          # Main ResearchGPT class
├── research_agents.py            # AI agents implementation
├── main.py                       # Main execution script
├── test_system.py                # Testing and evaluation
├── data/
│   ├── sample_papers/            # Sample PDF research papers
│   └── processed/                # Extracted text files
├── results/
│   ├── summaries/                # Generated summaries
│   ├── analyses/                 # Research analyses
│   └── evaluation_results.txt    # Performance metrics
└── prompts/
    └── prompt_templates.txt      # Prompt templates
```

## Installation

### Prerequisites
- Python 3.8 or higher
- Mistral API key
- Git

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/research-gpt-assistant.git
   cd research-gpt-assistant
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API settings**
   - Open `config.py`
   - Replace `"your-mistral-api-key-here"` with your actual Mistral API key
   - Adjust other configuration parameters as needed

5. **Create required directories**
   ```bash
   mkdir -p data/sample_papers data/processed results/summaries results/analyses prompts
   ```

6. **Add sample research papers**
   - Place PDF research papers in the `data/sample_papers/` directory
   - Ensure papers are relevant to your research interests

## Usage

### Basic Usage

1. **Run the main demonstration**
   ```bash
   python main.py
   ```
   This will:
   - Process all PDF files in the sample_papers directory
   - Build a searchable index of documents
   - Demonstrate all system capabilities
   - Save results to the results directory

2. **Run system tests**
   ```bash
   python test_system.py
   ```
   This will:
   - Test all system components
   - Evaluate performance metrics
   - Generate comprehensive evaluation report

### Advanced Usage

#### Custom Research Queries
```python
from config import Config
from document_processor import DocumentProcessor
from research_assistant import ResearchGPTAssistant

# Initialize system
config = Config()
doc_processor = DocumentProcessor(config)
assistant = ResearchGPTAssistant(config, doc_processor)

# Process documents and build index
doc_processor.process_document("path/to/your/paper.pdf")
doc_processor.build_search_index()

# Ask research questions
response = assistant.answer_research_question(
    "What are the main limitations of current machine learning approaches?",
    use_cot=True,
    use_verification=True
)

print(response['answer'])
```

#### Using AI Agents
```python
from research_agents import AgentOrchestrator

orchestrator = AgentOrchestrator(assistant)

# Summarize a document
summary = orchestrator.route_task('summarizer', {'doc_id': 'paper_id'})

# Answer analytical questions
qa_result = orchestrator.route_task('qa', {
    'question': 'How do transformer models work?',
    'type': 'analytical'
})

# Conduct complete research session
research_session = orchestrator.route_task('workflow', {
    'research_topic': 'natural language processing trends'
})
```

## Configuration

### API Settings
Edit `config.py` to customize:
- **MISTRAL_API_KEY**: Your Mistral API key
- **MODEL_NAME**: Mistral model to use (default: "mistral-medium")
- **TEMPERATURE**: Response randomness (0.0-1.0, default: 0.1)
- **MAX_TOKENS**: Maximum response length (default: 1000)

### Processing Parameters
- **CHUNK_SIZE**: Text chunk size for processing (default: 1000)
- **OVERLAP**: Overlap between text chunks (default: 100)

### Directory Paths
All data and result paths can be customized in the configuration file.

## Examples

### Example 1: Document Summarization
```python
# Process a research paper and generate summary
doc_id = doc_processor.process_document("data/sample_papers/ai_paper.pdf")
summary = agent_orchestrator.route_task('summarizer', {'doc_id': doc_id})
print(f"Summary: {summary['summary']}")
```

### Example 2: Chain-of-Thought Reasoning
```python
# Use advanced reasoning for complex questions
query = "Compare the effectiveness of supervised vs unsupervised learning"
response = assistant.chain_of_thought_reasoning(query, relevant_chunks)
print(f"CoT Response: {response}")
```

### Example 3: Research Workflow Automation
```python
# Automate complete research process
workflow = assistant.react_research_workflow(
    "What are the current challenges in deep learning?"
)
for step in workflow['workflow_steps']:
    print(f"Step {step['step']}: {step['thought']}")
print(f"Conclusion: {workflow['final_answer']}")
```

## Testing

### Running Tests
The system includes comprehensive testing capabilities:

```bash
# Run all tests
python test_system.py

# Check specific components
python -c "from test_system import ResearchGPTTester; tester = ResearchGPTTester(); tester.test_document_processing()"
```

### Test Coverage
- Document processing functionality
- Prompting strategy performance
- AI agent capabilities
- System integration and workflows
- Performance benchmarking

## Performance

### System Requirements
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 1GB free space for documents and results
- **Network**: Internet connection for Mistral API calls

### Performance Metrics
- Document processing: ~1-2 seconds per PDF page
- Query response time: 2-5 seconds depending on complexity
- API calls: Optimized to minimize usage while maintaining quality

## Evaluation

The system includes built-in evaluation metrics:
- Response relevance and quality scoring
- Processing speed benchmarks
- Strategy effectiveness comparison
- Agent performance analysis

Results are automatically saved to:
- `results/evaluation_report.md`: Comprehensive evaluation summary
- `results/test_results.json`: Detailed test metrics

## Troubleshooting

### Common Issues

1. **API Key Error**
   - Ensure your Mistral API key is correctly set in `config.py`
   - Verify the API key has sufficient credits

2. **PDF Processing Failed**
   - Check if PDF files are not password-protected
   - Ensure PDF files contain extractable text (not just images)

3. **Import Errors**
   - Verify all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility (3.8+)

4. **Empty Search Results**
   - Ensure documents are processed and indexed before querying
   - Check if query terms match document content

### Getting Help
- Check the evaluation report for system performance insights
- Review test results for component-specific issues
- Ensure all configuration settings are appropriate for your use case

## Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Submit a pull request with detailed description

### Code Style
- Follow PEP 8 Python style guidelines
- Include comprehensive docstrings and comments
- Add tests for new functionality
- Update documentation as needed

## License

This project is created for educational purposes as part of an AI/ML capstone project by Rama Kattunga.

## Acknowledgments

This project demonstrates practical application of:
- Machine Learning fundamentals and algorithms
- Natural Language Processing techniques
- Advanced prompting strategies for Large Language Models
- AI agent architecture and workflow automation
- Software engineering best practices for AI applications

## Contact

For questions or support regarding this educational project, please refer to your course materials or contact your instructor.

## Version History

- **v1.0.0**: Initial release with core functionality
  - Document processing and indexing
  - Basic prompting strategies
  - AI agent implementation
  - Testing and evaluation framework
