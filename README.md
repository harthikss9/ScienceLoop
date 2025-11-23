# ScienceLoop - AI4Science Project

An AI4Science project using SpoonOS for understanding scientific papers, generating hypotheses, creating simulation plans, generating executable code, running simulations, fixing errors, and generating comprehensive reports.

## Project Structure

```
ScienceLoop/
├── agents/
│   ├── PaperUnderstandingAgent.py    # Step 1: Analyzes PDFs and extracts structured data
│   ├── KnowledgeGraphAgent.py         # Step 2: Builds knowledge graph from Step 1 output
│   ├── HypothesisAgent.py             # Step 3: Generates testable hypotheses
│   ├── SimulationPlanAgent.py         # Step 4: Creates simulation-ready plans
│   ├── DatasetAgent.py                # Step 5: Generates/downloads required datasets
│   ├── CodeGeneratorAgent.py         # Step 6: Generates runnable Python code
│   ├── SimulationRunnerAgent.py       # Step 7: Executes simulations and captures results
│   ├── ErrorFeedbackAgent.py          # Step 8: Analyzes errors and generates fix requests
│   └── ReportAgent.py                 # Step 9: Generates comprehensive reports
├── tools/
│   └── pdf_reader_tool.py             # SpoonOS PDF reader tool (BaseTool)
├── papers/                            # Place PDF files here
│   ├── bio paper/                    # Example: Biology paper with all outputs
│   ├── sna paper/                    # Example: Network science paper with all outputs
│   ├── physics paper/                # Example: Physics paper with all outputs
│   ├── mldl/                         # Example: ML/DL paper with all outputs
│   └── saish/                        # Example: Classification paper with all outputs
├── results/                           # Simulation run results (auto-generated)
│   └── {run_id}/
│       ├── logs/
│       │   ├── stdout.txt
│       │   └── stderr.txt
│       ├── artifacts/                 # Generated plots, CSV files, etc.
│       ├── error_report.json          # (or simulation_result.json)
│       └── fix_request.json           # (if error fixing was performed)
├── spoon.json                        # SpoonOS configuration
├── requirements.txt                  # Python dependencies
├── AGENT_INPUTS.md                   # Complete agent inputs reference
└── README.md                         # This file
```

## Setup

1. **Create and activate virtual environment:**
```bash
# macOS/Linux
python3 -m venv spoon-env
source spoon-env/bin/activate

# Windows (PowerShell)
python -m venv spoon-env
.\spoon-env\Scripts\Activate.ps1
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure API keys in `.env` file:**

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
- **SimulationRunnerAgent**: Pure Python (no LLM)
- **ErrorFeedbackAgent**: Uses `gpt-4o` (via `OPENAI_MODEL`)
- **ReportAgent**: Uses `claude-sonnet-4-5-20250929` (via `ANTHROPIC_MODEL`)

4. **Place your PDF files in the `papers/` directory.**

## Complete Pipeline Overview

The ScienceLoop pipeline consists of **9 agents** working in sequence:

### **Phase 1: Paper Analysis (Steps 1-4)**
1. **PaperUnderstandingAgent**: Extracts formulas, variables, relationships, and key ideas from PDFs
2. **KnowledgeGraphAgent**: Creates a scientific knowledge graph with nodes and edges
3. **HypothesisAgent**: Generates testable scientific hypotheses
4. **SimulationPlanAgent**: Creates precise, executable simulation plans

### **Phase 2: Code & Data Preparation (Steps 5-6)**
5. **DatasetAgent**: Automatically generates/downloads required datasets
6. **CodeGeneratorAgent**: Converts simulation plans into runnable Python scripts

### **Phase 3: Simulation Execution (Step 7)**
7. **SimulationRunnerAgent**: Executes simulation scripts, captures outputs, stores artifacts

### **Phase 4: Error Fixing (Step 8 - Manual Loop)**
8. **ErrorFeedbackAgent**: Analyzes errors and generates fix requests for code regeneration

### **Phase 5: Report Generation (Step 9)**
9. **ReportAgent**: Generates comprehensive reports comparing expected vs actual outcomes

## Complete Pipeline Flow

```
PDF
  ↓
PaperUnderstandingAgent → understand.json
  ↓
KnowledgeGraphAgent → knowledge_graph.json
  ↓
HypothesisAgent → hypothesis.json
  ↓
SimulationPlanAgent → simulation_plan.json
  ↓                    ↓
DatasetAgent      CodeGeneratorAgent
  ↓                    ↓
datasets_manifest.json  simulation.py
  ↓                    ↓
SimulationRunnerAgent → error_report.json / simulation_result.json
  ↓ (if error)
ErrorFeedbackAgent → fix_request.json → CodeGeneratorAgent (regenerate)
  ↓ (repeat until success)
ReportAgent → report.json, report.md
```

## Usage

**Important:** Make sure the virtual environment is activated before running:

```bash
source spoon-env/bin/activate  # macOS/Linux
# or
.\spoon-env\Scripts\Activate.ps1  # Windows PowerShell
```

### Complete Pipeline Example

For a paper in `papers/saish/`:

#### **Phase 1: Paper Analysis**

```bash
# Step 1: Paper Understanding
python agents/PaperUnderstandingAgent.py --pdf "papers/saish/saish paper.pdf"
# Output: papers/saish/understand.json

# Step 2: Knowledge Graph
python agents/KnowledgeGraphAgent.py --input "papers/saish/understand.json"
# Output: papers/saish/knowledge_graph.json

# Step 3: Hypothesis Generation
python agents/HypothesisAgent.py --input "papers/saish/understand.json" --kg "papers/saish/knowledge_graph.json"
# Output: papers/saish/hypothesis.json

# Step 4: Simulation Plan
python agents/SimulationPlanAgent.py --hypothesis "papers/saish/hypothesis.json" --paper "papers/saish/understand.json" --kg "papers/saish/knowledge_graph.json"
# Output: papers/saish/simulation_plan.json
```

#### **Phase 2: Code & Data Preparation**

```bash
# Step 5: Dataset Generation
python agents/DatasetAgent.py --plan "papers/saish/simulation_plan.json"
# Output: papers/saish/datasets_manifest.json + datasets/

# Step 6: Code Generation (with datasets)
python agents/CodeGeneratorAgent.py --plan "papers/saish/simulation_plan.json" --datasets "papers/saish/datasets_manifest.json"
# Output: papers/saish/simulation.py
```

#### **Phase 3: Simulation Execution**

```bash
# Step 7: Run Simulation
python agents/SimulationRunnerAgent.py \
  --python-file "papers/saish/simulation.py" \
  --datasets "papers/saish/datasets_manifest.json" \
  --simulation-plan "papers/saish/simulation_plan.json" \
  --timeout 600
# Output: results/{run_id}/error_report.json or simulation_result.json
```

#### **Phase 4: Error Fixing Loop (Manual - Repeat Until Success)**

If the simulation has errors, fix them manually:

```bash
# Get latest error report
LATEST_RESULT=$(ls -td results/*/error_report.json 2>/dev/null | head -1)
echo "Error report: $LATEST_RESULT"

# Generate fix request
python agents/ErrorFeedbackAgent.py \
  --error-report "$LATEST_RESULT" \
  --simulation-plan "papers/saish/simulation_plan.json" \
  --original-code "papers/saish/simulation.py" \
  --output "$(dirname $LATEST_RESULT)/fix_request.json"

# Regenerate code with fixes
TEMP_PLAN=$(python3 -c "
import json
import tempfile
f = open('$(dirname $LATEST_RESULT)/fix_request.json')
d = json.load(f)
p = d['simulation_plan'].copy()
p.update({
    '_fix_instructions': d['fix_instructions'],
    '_error_context': d['error_context'],
    '_error_summary': d['error_summary'],
    '_explanation': d['explanation']
})
t = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
json.dump(p, t, indent=2)
t.close()
print(t.name)
")
python agents/CodeGeneratorAgent.py --plan "$TEMP_PLAN" --datasets "papers/saish/datasets_manifest.json"
rm "$TEMP_PLAN"

# Re-run simulation (repeat Step 7)
python agents/SimulationRunnerAgent.py \
  --python-file "papers/saish/simulation.py" \
  --datasets "papers/saish/datasets_manifest.json" \
  --simulation-plan "papers/saish/simulation_plan.json" \
  --timeout 600
```

**Repeat Phase 4 until simulation succeeds (status: "success")**

#### **Phase 5: Report Generation (Only After Success)**

```bash
# Step 9: Generate Report
LATEST_RESULT=$(ls -td results/*/simulation_result.json 2>/dev/null | head -1)
python agents/ReportAgent.py \
  --paper-understanding "papers/saish/understand.json" \
  --knowledge-graph "papers/saish/knowledge_graph.json" \
  --hypothesis "papers/saish/hypothesis.json" \
  --simulation-plan "papers/saish/simulation_plan.json" \
  --simulation-result "$LATEST_RESULT" \
  --output "papers/saish/report.json" \
  --markdown-output "papers/saish/report.md"
# Output: papers/saish/report.json, papers/saish/report.md
```

## Agent Inputs Reference

### Quick Reference Table

| Agent | Required Inputs | Optional Inputs |
|-------|----------------|-----------------|
| **PaperUnderstandingAgent** | `--pdf` | None |
| **KnowledgeGraphAgent** | None (uses `--input` or stdin) | `--input` |
| **HypothesisAgent** | `--input`, `--kg` | None |
| **SimulationPlanAgent** | `--hypothesis`, `--paper`, `--kg` | None |
| **DatasetAgent** | `--plan` | None |
| **CodeGeneratorAgent** | `--plan` | `--datasets` |
| **SimulationRunnerAgent** | `--python-file` | `--datasets`, `--workdir`, `--timeout`, `--auto-fix`, `--simulation-plan` |
| **ErrorFeedbackAgent** | `--error-report`, `--simulation-plan` | `--original-code`, `--output` |
| **ReportAgent** | `--paper-understanding`, `--knowledge-graph`, `--hypothesis`, `--simulation-plan`, `--simulation-result` | `--error-history`, `--output`, `--markdown-output` |

### Detailed Agent Specifications

#### 1. **PaperUnderstandingAgent**
- **Input**: `--pdf` (string): Path to PDF file
- **Output**: `understand.json` (saved in PDF's directory)
- **Model**: GPT-4 Turbo
- **Purpose**: Extracts structured information from scientific PDFs

#### 2. **KnowledgeGraphAgent**
- **Input**: `--input` (string, optional): Path to understand.json (or reads from stdin)
- **Output**: `knowledge_graph.json` (saved in same directory)
- **Model**: GPT-4 Turbo
- **Purpose**: Builds scientific knowledge graph

#### 3. **HypothesisAgent**
- **Input**: 
  - `--input` (string): Path to understand.json
  - `--kg` (string): Path to knowledge_graph.json
- **Output**: `hypothesis.json` (saved in same directory)
- **Model**: Claude 4.5 Sonnet
- **Purpose**: Generates testable scientific hypotheses

#### 4. **SimulationPlanAgent**
- **Input**:
  - `--hypothesis` (string): Path to hypothesis.json
  - `--paper` (string): Path to understand.json
  - `--kg` (string): Path to knowledge_graph.json
- **Output**: `simulation_plan.json` (saved in same directory)
- **Model**: Claude 4.5 Sonnet
- **Purpose**: Creates executable simulation plans

#### 5. **DatasetAgent**
- **Input**: `--plan` (string): Path to simulation_plan.json
- **Output**: 
  - `datasets_manifest.json` (saved in same directory)
  - Dataset files in `datasets/` subdirectory
- **Model**: Claude 4.5 Sonnet
- **Purpose**: Generates/downloads required datasets

#### 6. **CodeGeneratorAgent**
- **Input**:
  - `--plan` (string): Path to simulation_plan.json
  - `--datasets` (string, optional): Path to datasets_manifest.json (recommended)
- **Output**: `simulation.py` (saved in same directory)
- **Model**: GPT-4o
- **Purpose**: Generates runnable Python simulation code
- **Features**:
  - Intelligently infers required libraries
  - Generates complete, standalone scripts
  - Handles error fix instructions from ErrorFeedbackAgent
  - Converts dataset paths to be relative to script directory
  - Validates code before returning

#### 7. **SimulationRunnerAgent**
- **Input**:
  - `--python-file` (string): Path to simulation.py
  - `--datasets` (string, optional): Path to datasets_manifest.json
  - `--workdir` (string, optional): Working directory (default: parent of python-file)
  - `--timeout` (number, optional): Timeout in seconds (default: 600)
  - `--auto-fix` (boolean, optional): Auto-fix errors (requires --simulation-plan)
  - `--simulation-plan` (string, optional): Required if --auto-fix is used
- **Output**: 
  - `results/{run_id}/error_report.json` (on error) or `simulation_result.json` (on success)
  - `results/{run_id}/logs/stdout.txt`
  - `results/{run_id}/logs/stderr.txt`
  - `results/{run_id}/artifacts/*` (plots, CSV files, etc.)
- **Model**: Pure Python (no LLM)
- **Purpose**: Executes simulations and captures all outputs

#### 8. **ErrorFeedbackAgent**
- **Input**:
  - `--error-report` (string): Path to error_report.json from SimulationRunnerAgent
  - `--simulation-plan` (string): Path to simulation_plan.json
  - `--original-code` (string, optional): Path to simulation.py
  - `--output` (string, optional): Path to save fix_request.json
- **Output**: `fix_request.json` (saved to specified path or printed to stdout)
- **Model**: GPT-4o
- **Purpose**: Analyzes errors and generates structured fix requests
- **Features**:
  - Detects specific error types (FileNotFoundError, ValueError, dimension mismatches)
  - Provides actionable fix instructions
  - Includes error context and explanation

#### 9. **ReportAgent**
- **Input**:
  - `--paper-understanding` (string): Path to understand.json
  - `--knowledge-graph` (string): Path to knowledge_graph.json
  - `--hypothesis` (string): Path to hypothesis.json
  - `--simulation-plan` (string): Path to simulation_plan.json
  - `--simulation-result` (string): Path to error_report.json or simulation_result.json
  - `--error-history` (string, optional): Path to JSON list of error reports
  - `--output` (string, optional): Path to save report.json
  - `--markdown-output` (string, optional): Path to save report.md
- **Output**: 
  - `report.json` (machine-readable summary)
  - `report.md` (human-readable report)
- **Model**: Claude 4.5 Sonnet
- **Purpose**: Generates comprehensive analysis reports
- **Note**: Only run after simulation succeeds (status: "success")

## Output Files

All outputs are **dynamically saved** in the same directory as the input PDF:

```
papers/your-paper/
├── paper.pdf                    # Original PDF
├── understand.json              # Step 1: Paper understanding
├── knowledge_graph.json         # Step 2: Knowledge graph
├── hypothesis.json              # Step 3: Hypothesis
├── simulation_plan.json         # Step 4: Simulation plan
├── datasets_manifest.json       # Step 5: Dataset manifest
├── datasets/                    # Step 5: Generated datasets
│   ├── synthetic/
│   │   └── classification_data.csv
│   └── ...
├── simulation.py               # Step 6: Generated Python code
└── report.json                 # Step 9: Report (JSON)
└── report.md                   # Step 9: Report (Markdown)

results/{run_id}/
├── logs/
│   ├── stdout.txt
│   └── stderr.txt
├── artifacts/                  # Generated plots, CSV files, etc.
│   ├── plot1.png
│   └── ...
├── error_report.json           # (on error) or simulation_result.json (on success)
└── fix_request.json            # (if error fixing was performed)
```

## Output Formats

### understand.json
```json
{
  "summary": "Brief scientific summary",
  "formulas": ["formula 1", "formula 2"],
  "relationships": ["relationship 1", "relationship 2"],
  "variables": ["variable 1: description", "variable 2: description"],
  "key_ideas": ["idea 1", "idea 2"]
}
```

### knowledge_graph.json
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

### hypothesis.json
```json
{
  "hypothesis": "A clear, testable scientific hypothesis statement",
  "justification": "Explanation of why this hypothesis follows logically from the paper"
}
```

### simulation_plan.json
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

### datasets_manifest.json
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

### error_report.json / simulation_result.json
```json
{
  "status": "error" | "success",
  "run_id": "<uuid>",
  "stdout": "<full text log>",
  "stderr": "<full text log or empty>",
  "exit_code": <int>,
  "artifacts": [
    {
      "filename": "plot1.png",
      "path": "results/<run_id>/artifacts/plot1.png"
    }
  ],
  "results_path": "results/<run_id>",
  "error_summary": "<if error>" (optional)
}
```

### fix_request.json
```json
{
  "needs_regeneration": true,
  "error_context": "<exact traceback>",
  "explanation": "<why the error occurred>",
  "fix_instructions": [
    "Correct variable X",
    "Import Y library",
    "Adjust dataset path to Z"
  ],
  "simulation_plan": { ... }
}
```

### report.json
```json
{
  "success": true | false | "partial",
  "reason": "<one-paragraph explanation>",
  "matched_expectations": [
    "<which parts of expected_outcomes were clearly observed>"
  ],
  "unmet_expectations": [
    "<which expected behaviors were not clearly observed or contradicted>"
  ],
  "key_observations": [
    "<short bullet strings for main findings>"
  ],
  "recommendations": [
    "<concrete next actions for future runs or code changes>"
  ]
}
```

## Example Workflows

### Complete Pipeline for Biology Paper
```bash
# Phase 1: Analysis
python agents/PaperUnderstandingAgent.py --pdf "papers/bio paper/scalley97A.pdf"
python agents/KnowledgeGraphAgent.py --input "papers/bio paper/understand.json"
python agents/HypothesisAgent.py --input "papers/bio paper/understand.json" --kg "papers/bio paper/knowledge_graph.json"
python agents/SimulationPlanAgent.py --hypothesis "papers/bio paper/hypothesis.json" --paper "papers/bio paper/understand.json" --kg "papers/bio paper/knowledge_graph.json"

# Phase 2: Preparation
python agents/DatasetAgent.py --plan "papers/bio paper/simulation_plan.json"
python agents/CodeGeneratorAgent.py --plan "papers/bio paper/simulation_plan.json" --datasets "papers/bio paper/datasets_manifest.json"

# Phase 3: Execution
python agents/SimulationRunnerAgent.py --python-file "papers/bio paper/simulation.py" --datasets "papers/bio paper/datasets_manifest.json" --simulation-plan "papers/bio paper/simulation_plan.json"

# Phase 4: Error Fixing (if needed - repeat until success)
# ... (see manual error fixing loop above)

# Phase 5: Report (only after success)
LATEST_RESULT=$(ls -td results/*/simulation_result.json 2>/dev/null | head -1)
python agents/ReportAgent.py --paper-understanding "papers/bio paper/understand.json" --knowledge-graph "papers/bio paper/knowledge_graph.json" --hypothesis "papers/bio paper/hypothesis.json" --simulation-plan "papers/bio paper/simulation_plan.json" --simulation-result "$LATEST_RESULT" --output "papers/bio paper/report.json" --markdown-output "papers/bio paper/report.md"
```

## Error Handling

### Manual Error Fixing Loop

When `SimulationRunnerAgent` reports an error:

1. **Get the latest error report:**
   ```bash
   LATEST_RESULT=$(ls -td results/*/error_report.json 2>/dev/null | head -1)
   ```

2. **Generate fix request:**
   ```bash
   python agents/ErrorFeedbackAgent.py \
     --error-report "$LATEST_RESULT" \
     --simulation-plan "papers/your-paper/simulation_plan.json" \
     --original-code "papers/your-paper/simulation.py" \
     --output "$(dirname $LATEST_RESULT)/fix_request.json"
   ```

3. **Regenerate code with fixes:**
   ```bash
   TEMP_PLAN=$(python3 -c "
   import json
   import tempfile
   f = open('$(dirname $LATEST_RESULT)/fix_request.json')
   d = json.load(f)
   p = d['simulation_plan'].copy()
   p.update({
       '_fix_instructions': d['fix_instructions'],
       '_error_context': d['error_context'],
       '_error_summary': d['error_summary'],
       '_explanation': d['explanation']
   })
   t = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
   json.dump(p, t, indent=2)
   t.close()
   print(t.name)
   ")
   python agents/CodeGeneratorAgent.py --plan "$TEMP_PLAN" --datasets "papers/your-paper/datasets_manifest.json"
   rm "$TEMP_PLAN"
   ```

4. **Re-run simulation:**
   ```bash
   python agents/SimulationRunnerAgent.py \
     --python-file "papers/your-paper/simulation.py" \
     --datasets "papers/your-paper/datasets_manifest.json" \
     --simulation-plan "papers/your-paper/simulation_plan.json"
   ```

5. **Repeat steps 1-4 until status is "success"**

### Common Errors & Solutions

1. **`FileNotFoundError: [Errno 2] No such file or directory: 'papers/.../datasets/...'`**
   - **Solution**: `CodeGeneratorAgent` now converts dataset paths to be relative to the script's directory. Ensure `--datasets` argument is provided.

2. **`ValueError: x and y must have same first dimension`**
   - **Solution**: `ErrorFeedbackAgent` detects this and provides specific fix instructions. Regenerate code with the fix request.

3. **`ModuleNotFoundError: No module named 'X'`**
   - **Solution**: Install missing dependencies: `pip install -r requirements.txt`

4. **Simulation hangs or times out**
   - **Solution**: Increase `--timeout` value or optimize simulation code (reduce parameter ranges, add progress output)

5. **Report generation fails**
   - **Solution**: Ensure simulation completed successfully first (status: "success" in simulation_result.json)

## Dependencies

See `requirements.txt` for full list. Key dependencies:

- `spoon-ai-sdk>=0.3.0` - SpoonOS core framework
- `spoon-toolkits>=0.2.0` - Extended toolkits
- `pypdf>=3.0.0` - PDF reading
- `openai>=1.0.0` - OpenAI API client
- `anthropic>=0.34.0` - Anthropic/Claude API client
- `networkx>=3.0` - Graph generation
- `numpy>=1.24.0` - Numerical computations
- `scipy>=1.10.0` - Scientific computing
- `matplotlib>=3.7.0` - Plotting
- `scikit-learn>=1.3.0` - Machine learning
- `pandas` - Data manipulation
- `requests>=2.31.0` - HTTP requests
- `python-dotenv>=1.0.0` - Environment variable management

## Key Features

- **Dynamic File Saving**: All outputs are saved in the same directory as the input PDF
- **Domain Agnostic**: Works for biology, chemistry, physics, computer science, ML, algorithms, systems, etc.
- **Model Selection**: Uses GPT-4 Turbo for understanding/graphs, Claude 4.5 Sonnet for reasoning/planning, GPT-4o for code generation
- **Error Handling**: Robust error detection, analysis, and automated code regeneration
- **Code Validation**: Pre-generation validation to catch common errors before runtime
- **Intelligent Plotting**: CodeGeneratorAgent intelligently decides when to generate plots based on simulation plan
- **Comprehensive Reports**: Detailed analysis comparing expected vs actual outcomes

## Notes

- **Context Windows**: Agents handle large PDFs with truncation and model fallbacks
- **Error Recovery**: Manual error-fixing loop ensures robust code generation
- **Path Handling**: All dataset paths are automatically converted to be relative to script execution directory
- **Plot Saving**: Generated code uses `plt.savefig()` and `plt.close()` instead of `plt.show()` to prevent blocking
- **Non-Interactive Backend**: Matplotlib uses 'Agg' backend for non-interactive plotting

## Additional Resources

- **`AGENT_INPUTS.md`**: Complete reference for all agent inputs and usage
- **`RUN_SAISH_PAPER.md`**: Step-by-step guide for running complete pipeline
- **`MANUAL_ERROR_FIX.md`**: Detailed error fixing instructions

## License

MIT License
