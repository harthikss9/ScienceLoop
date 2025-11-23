# ScienceLoop - AI4Science Project

An AI4Science project using SpoonOS for understanding scientific papers, generating hypotheses, creating simulation plans, and generating executable code.

## Project Structure

```
ScienceLoop/
├── agents/
│   ├── PaperUnderstandingAgent.py    # Step 1: Analyzes PDFs and extracts structured data
│   ├── KnowledgeGraphAgent.py        # Step 2: Builds knowledge graph from Step 1 output
│   ├── HypothesisAgent.py            # Step 3: Generates testable hypotheses
│   ├── SimulationPlanAgent.py        # Step 4: Creates simulation-ready plans
│   ├── CodeGeneratorAgent.py         # Step 5: Generates runnable Python code
│   └── DatasetAgent.py               # Step 6: Generates/downloads required datasets
├── tools/
│   └── pdf_reader_tool.py            # SpoonOS PDF reader tool (BaseTool)
├── papers/                            # Place PDF files here
│   ├── bio paper/                    # Example: Biology paper with all outputs
│   └── sna paper/                    # Example: Network science paper with all outputs
├── spoon.json                        # SpoonOS configuration with MCP servers
├── requirements.txt                  # Python dependencies
└── workflow.sh                       # Complete workflow script
```

## Setup

1. Create and activate virtual environment:
```bash
# macOS/Linux
python3 -m venv spoon-env
source spoon-env/bin/activate

# Windows (PowerShell)
python -m venv spoon-env
.\spoon-env\Scripts\Activate.ps1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API keys in `.env` file:

Create a `.env` file in the project root:
```bash
# OpenAI API Configuration
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-4-turbo

# Anthropic/Claude API Configuration
ANTHROPIC_API_KEY=your-anthropic-api-key-here
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929
```

**Model Configuration:**
- **PaperUnderstandingAgent**: Uses `gpt-4-turbo` (via `OPENAI_MODEL`)
- **KnowledgeGraphAgent**: Uses `gpt-4-turbo` (via `OPENAI_MODEL`)
- **HypothesisAgent**: Uses `claude-sonnet-4-5-20250929` (via `ANTHROPIC_MODEL`)
- **SimulationPlanAgent**: Uses `claude-sonnet-4-5-20250929` (via `ANTHROPIC_MODEL`)
- **CodeGeneratorAgent**: Uses `gpt-4o` (via `OPENAI_MODEL`, best for code generation)
- **DatasetAgent**: Uses `claude-sonnet-4-5-20250929` (via `ANTHROPIC_MODEL`)

4. Place your PDF files in the `papers/` directory.

## MCP Integration

The project includes MCP (Model Context Protocol) servers for enhanced tool access:

- **filesystem**: File operations (`readFile`, `writeFile`, `mkdir`, `list`)
- **process**: Process execution (`run`)
- **shell**: Shell commands (`run`)

These are configured in `spoon.json` and available to all agents through the Spoon MCP protocol.

## Complete Workflow

The ScienceLoop pipeline consists of 6 agents that work together:

### Step 1: Paper Understanding
Extracts formulas, variables, relationships, and key ideas from PDFs.

### Step 2: Knowledge Graph Building
Creates a scientific knowledge graph with nodes and edges.

### Step 3: Hypothesis Generation
Generates testable scientific hypotheses using Claude 4.5 Sonnet.

### Step 4: Simulation Planning
Creates precise, executable simulation plans.

### Step 5: Code Generation
Converts simulation plans into runnable Python scripts using GPT-4o.

### Step 6: Dataset Generation
Automatically generates/downloads required datasets (graphs, ML data, bio structures).

## Usage

**Important:** Make sure the virtual environment is activated before running:

```bash
source spoon-env/bin/activate  # macOS/Linux
# or
.\spoon-env\Scripts\Activate.ps1  # Windows PowerShell
```

### Complete Pipeline (All 6 Agents)

Run all agents in sequence for a paper:

```bash
# One-liner (all at once)
source spoon-env/bin/activate && \
python agents/PaperUnderstandingAgent.py --pdf "papers/your-paper/paper.pdf" >/dev/null 2>&1 && \
python agents/KnowledgeGraphAgent.py --input "papers/your-paper/understand.json" >/dev/null 2>&1 && \
python agents/HypothesisAgent.py --input "papers/your-paper/understand.json" --kg "papers/your-paper/knowledge_graph.json" >/dev/null 2>&1 && \
python agents/SimulationPlanAgent.py --hypothesis "papers/your-paper/hypothesis.json" --paper "papers/your-paper/understand.json" --kg "papers/your-paper/knowledge_graph.json" >/dev/null 2>&1 && \
python agents/CodeGeneratorAgent.py --plan "papers/your-paper/simulation_plan.json" >/dev/null 2>&1 && \
python agents/DatasetAgent.py --plan "papers/your-paper/simulation_plan.json" && \
echo "✅ All done! Check papers/your-paper/ for all outputs"
```

### Step-by-Step Execution

```bash
# Activate virtual environment
source spoon-env/bin/activate

# Step 1: Paper Understanding
python agents/PaperUnderstandingAgent.py --pdf "papers/your-paper/paper.pdf"
# Output: papers/your-paper/understand.json

# Step 2: Knowledge Graph
python agents/KnowledgeGraphAgent.py --input "papers/your-paper/understand.json"
# Output: papers/your-paper/knowledge_graph.json

# Step 3: Hypothesis Generation
python agents/HypothesisAgent.py --input "papers/your-paper/understand.json" --kg "papers/your-paper/knowledge_graph.json"
# Output: papers/your-paper/hypothesis.json

# Step 4: Simulation Plan
python agents/SimulationPlanAgent.py --hypothesis "papers/your-paper/hypothesis.json" --paper "papers/your-paper/understand.json" --kg "papers/your-paper/knowledge_graph.json"
# Output: papers/your-paper/simulation_plan.json

# Step 5: Code Generation
python agents/CodeGeneratorAgent.py --plan "papers/your-paper/simulation_plan.json"
# Output: papers/your-paper/simulation.py

# Step 6: Dataset Generation
python agents/DatasetAgent.py --plan "papers/your-paper/simulation_plan.json"
# Output: papers/your-paper/datasets/ and papers/your-paper/datasets_manifest.json
```

### Using SpoonOS CLI

```bash
# Step 1
spoon run PaperUnderstandingAgent --pdf papers/your-paper/paper.pdf

# Step 2
spoon run KnowledgeGraphAgent --input papers/your-paper/understand.json

# Step 3
spoon run HypothesisAgent --input papers/your-paper/understand.json --kg papers/your-paper/knowledge_graph.json

# Step 4
spoon run SimulationPlanAgent --hypothesis papers/your-paper/hypothesis.json --paper papers/your-paper/understand.json --kg papers/your-paper/knowledge_graph.json

# Step 5
spoon run CodeGeneratorAgent --plan papers/your-paper/simulation_plan.json

# Step 6
spoon run DatasetAgent --plan papers/your-paper/simulation_plan.json
```

## Output Files

All outputs are **dynamically saved** in the same directory as the input PDF:

```
papers/your-paper/
├── paper.pdf                    # Original PDF
├── understand.json              # Step 1: Paper understanding
├── knowledge_graph.json         # Step 2: Knowledge graph
├── hypothesis.json              # Step 3: Hypothesis
├── simulation_plan.json          # Step 4: Simulation plan
├── simulation.py                # Step 5: Generated Python code
├── datasets_manifest.json       # Step 6: Dataset manifest
└── datasets/                    # Step 6: Generated datasets
    ├── barabasi_albert_n100_m3.edgelist
    ├── watts_strogatz_n100_k6_p0.30.edgelist
    ├── erdos_renyi_n100_p0.01.edgelist
    ├── karate.edgelist
    └── dolphins.edgelist
```

## Agent Details

### 1. PaperUnderstandingAgent

**Purpose**: Analyzes scientific PDFs to extract structured information.

**Input**: PDF file path

**Output**: JSON with:
- `summary`: Brief scientific summary
- `formulas`: List of equations/formulas
- `relationships`: Scientific relationships
- `variables`: Variables with descriptions
- `key_ideas`: Main scientific concepts

**Model**: GPT-4 Turbo (OpenAI)

**Features**:
- Raw PDF text extraction (no manual parsing)
- LLM understands formulas, variables, relationships automatically
- Domain-agnostic (works for any scientific field)
- Saves output as `understand.json` in PDF's directory

### 2. KnowledgeGraphAgent

**Purpose**: Builds a scientific knowledge graph from paper understanding.

**Input**: `understand.json` from Step 1

**Output**: JSON with:
- `nodes`: List of scientific entities
- `edges`: Relationships between entities (source, relation, target)

**Model**: GPT-4 Turbo (OpenAI)

**Features**:
- Extracts entities and relationships
- Creates directional knowledge graph
- Descriptive entity names (e.g., "Temperature" not "T")
- Saves output as `knowledge_graph.json` in same directory

### 3. HypothesisAgent

**Purpose**: Generates testable scientific hypotheses.

**Input**: 
- `understand.json` (Step 1 output)
- `knowledge_graph.json` (Step 2 output)

**Output**: JSON with:
- `hypothesis`: Clear, testable hypothesis statement
- `justification`: Why this hypothesis follows from the paper

**Model**: Claude 4.5 Sonnet (Anthropic)

**Features**:
- Grounded strictly in paper content (no hallucination)
- Specific and measurable
- Simulation-friendly
- Works for all scientific fields

### 4. SimulationPlanAgent

**Purpose**: Creates precise, executable simulation plans.

**Input**:
- `hypothesis.json` (Step 3 output)
- `understand.json` (Step 1 output)
- `knowledge_graph.json` (Step 2 output)

**Output**: JSON with:
- `simulation_equations`: Equations to implement
- `constants_required`: Constants with values/ranges
- `variables_to_vary`: Parameters to sweep
- `procedure_steps`: Step-by-step implementation plan
- `expected_outcomes`: What patterns to expect

**Model**: Claude 4.5 Sonnet (Anthropic)

**Features**:
- Executable in Python (NumPy/SciPy/Matplotlib)
- No ambiguity or guesswork
- Domain-agnostic

### 5. CodeGeneratorAgent

**Purpose**: Converts simulation plans into runnable Python code.

**Input**: `simulation_plan.json` (Step 4 output)

**Output**: 
- JSON with `python_code` field
- Saves `simulation.py` file

**Model**: GPT-4o (OpenAI, best for code generation)

**Features**:
- Intelligently infers required libraries (numpy, scipy, matplotlib, networkx, etc.)
- Generates complete, standalone Python scripts
- No placeholders or pseudocode
- Modular, readable code with docstrings
- Domain-agnostic

### 6. DatasetAgent

**Purpose**: Generates/downloads datasets required for simulation code.

**Input**: `simulation_plan.json` (Step 4 output)

**Output**: 
- JSON with `dataset_type` and `datasets` (name -> path mapping)
- Saves `datasets_manifest.json`
- Generates/downloads actual dataset files

**Model**: Claude 4.5 Sonnet (Anthropic)

**Features**:
- Auto-detects dataset type (graph, ml, bio_structures, none)
- Generates graph datasets:
  - Barabási–Albert (scale-free)
  - Watts–Strogatz (small-world)
  - Erdős–Rényi (random)
  - Real-world networks (karate, dolphins)
- Downloads PDB files for bio structures
- Creates ML dataset placeholders
- Saves all datasets to `datasets/` directory

## Output Formats

### Step 1: understand.json
```json
{
  "summary": "Brief scientific summary",
  "formulas": ["formula 1", "formula 2"],
  "relationships": ["relationship 1", "relationship 2"],
  "variables": ["variable 1: description", "variable 2: description"],
  "key_ideas": ["idea 1", "idea 2"]
}
```

### Step 2: knowledge_graph.json
```json
{
  "nodes": ["entity1", "entity2", "entity3"],
  "edges": [
    {
      "source": "entity1",
      "relation": "relationship description",
      "target": "entity2"
    }
  ]
}
```

### Step 3: hypothesis.json
```json
{
  "hypothesis": "A clear, testable scientific hypothesis statement",
  "justification": "Explanation of why this hypothesis follows logically from the paper"
}
```

### Step 4: simulation_plan.json
```json
{
  "simulation_equations": ["equation1", "equation2"],
  "constants_required": [
    {
      "name": "constant_name",
      "description": "what it represents",
      "value_or_range": "value or range"
    }
  ],
  "variables_to_vary": [
    {
      "name": "variable_name",
      "description": "what it represents",
      "range": "[min, max]",
      "units": "if applicable"
    }
  ],
  "procedure_steps": ["Step 1: ...", "Step 2: ..."],
  "expected_outcomes": "Description of expected patterns"
}
```

### Step 5: simulation.py
Complete, runnable Python script with:
- All necessary imports
- Constant initialization
- Functions for equations
- Variable sweeps
- Plotting functions
- Main execution function

### Step 6: datasets_manifest.json
```json
{
  "dataset_type": "graph",
  "datasets": {
    "barabasi_albert": "path/to/barabasi_albert_n100_m3.edgelist",
    "watts_strogatz": "path/to/watts_strogatz_n100_k6_p0.30.edgelist"
  },
  "reasoning": "Explanation of dataset requirements"
}
```

## Example Workflows

### Biology Paper
```bash
python agents/PaperUnderstandingAgent.py --pdf "papers/bio paper/scalley97A.pdf"
python agents/KnowledgeGraphAgent.py --input "papers/bio paper/understand.json"
python agents/HypothesisAgent.py --input "papers/bio paper/understand.json" --kg "papers/bio paper/knowledge_graph.json"
python agents/SimulationPlanAgent.py --hypothesis "papers/bio paper/hypothesis.json" --paper "papers/bio paper/understand.json" --kg "papers/bio paper/knowledge_graph.json"
python agents/CodeGeneratorAgent.py --plan "papers/bio paper/simulation_plan.json"
python agents/DatasetAgent.py --plan "papers/bio paper/simulation_plan.json"
```

### Network Science Paper
```bash
python agents/PaperUnderstandingAgent.py --pdf "papers/sna paper/Complexity - 2020 - Zhao - Ranking Influential Nodes in Complex Networks with Information Entropy Method.pdf"
python agents/KnowledgeGraphAgent.py --input "papers/sna paper/understand.json"
python agents/HypothesisAgent.py --input "papers/sna paper/understand.json" --kg "papers/sna paper/knowledge_graph.json"
python agents/SimulationPlanAgent.py --hypothesis "papers/sna paper/hypothesis.json" --paper "papers/sna paper/understand.json" --kg "papers/sna paper/knowledge_graph.json"
python agents/CodeGeneratorAgent.py --plan "papers/sna paper/simulation_plan.json"
python agents/DatasetAgent.py --plan "papers/sna paper/simulation_plan.json"
```

## Running Generated Simulations

After completing all 6 steps, you can run the generated simulation:

```bash
cd papers/your-paper/
python simulation.py
```

The simulation code will automatically use the datasets from the `datasets/` directory.

## Dependencies

See `requirements.txt` for full list. Key dependencies:

- `spoon-ai-sdk>=0.3.0` - SpoonOS core framework
- `spoon-toolkits>=0.2.0` - Extended toolkits
- `PyPDF2>=3.0.0` - PDF reading
- `openai>=1.0.0` - OpenAI API client
- `anthropic>=0.34.0` - Anthropic/Claude API client
- `networkx>=3.0` - Graph generation
- `numpy>=1.24.0` - Numerical computations
- `requests>=2.31.0` - HTTP requests
- `python-dotenv>=1.0.0` - Environment variable management

## Notes

- **Dynamic File Saving**: All outputs are saved in the same directory as the input PDF
- **Domain Agnostic**: Works for biology, chemistry, physics, computer science, ML, algorithms, systems, etc.
- **Model Selection**: Uses GPT-4 Turbo for understanding/graphs, Claude 4.5 Sonnet for reasoning/planning, GPT-4o for code generation
- **Context Windows**: Agents handle large PDFs with truncation and model fallbacks
- **Error Handling**: Robust error handling with model fallbacks and validation

## License

MIT License
