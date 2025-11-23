# Agent Inputs Reference - Sanity Check

## Complete Pipeline Input Requirements

### **Phase 1: Paper Analysis**

#### 1. **PaperUnderstandingAgent**
**Required:**
- `--pdf` (string): Path to PDF file

**Example:**
```bash
python agents/PaperUnderstandingAgent.py --pdf "papers/saish/saish paper.pdf"
```

**Output:** `papers/{paper_name}/understand.json`

---

#### 2. **KnowledgeGraphAgent**
**Required:**
- `--input` (string, optional): Path to understand.json (if not provided, reads from stdin)

**Example:**
```bash
python agents/KnowledgeGraphAgent.py --input "papers/saish/understand.json"
```

**Output:** `papers/{paper_name}/knowledge_graph.json`

---

#### 3. **HypothesisAgent**
**Required:**
- `--input` (string): Path to understand.json
- `--kg` (string): Path to knowledge_graph.json

**Example:**
```bash
python agents/HypothesisAgent.py --input "papers/saish/understand.json" --kg "papers/saish/knowledge_graph.json"
```

**Output:** `papers/{paper_name}/hypothesis.json`

---

#### 4. **SimulationPlanAgent**
**Required:**
- `--hypothesis` (string): Path to hypothesis.json
- `--paper` (string): Path to understand.json
- `--kg` (string): Path to knowledge_graph.json

**Example:**
```bash
python agents/SimulationPlanAgent.py --hypothesis "papers/saish/hypothesis.json" --paper "papers/saish/understand.json" --kg "papers/saish/knowledge_graph.json"
```

**Output:** `papers/{paper_name}/simulation_plan.json`

---

### **Phase 2: Code & Data Preparation**

#### 5. **DatasetAgent**
**Required:**
- `--plan` (string): Path to simulation_plan.json

**Example:**
```bash
python agents/DatasetAgent.py --plan "papers/saish/simulation_plan.json"
```

**Output:** 
- `papers/{paper_name}/datasets_manifest.json`
- Dataset files in `papers/{paper_name}/datasets/`

---

#### 6. **CodeGeneratorAgent**
**Required:**
- `--plan` (string): Path to simulation_plan.json

**Optional:**
- `--datasets` (string): Path to datasets_manifest.json (recommended)

**Example:**
```bash
python agents/CodeGeneratorAgent.py --plan "papers/saish/simulation_plan.json" --datasets "papers/saish/datasets_manifest.json"
```

**Output:** `papers/{paper_name}/simulation.py`

---

### **Phase 3: Simulation Execution**

#### 7. **SimulationRunnerAgent**
**Required:**
- `--python-file` (string): Path to simulation.py

**Optional:**
- `--datasets` (string): Path to datasets_manifest.json
- `--workdir` (string): Working directory (default: parent of python-file)
- `--timeout` (number): Timeout in seconds (default: 600)
- `--auto-fix` (boolean): Auto-fix errors (requires --simulation-plan)
- `--simulation-plan` (string): Required if --auto-fix is used

**Example:**
```bash
python agents/SimulationRunnerAgent.py \
  --python-file "papers/saish/simulation.py" \
  --datasets "papers/saish/datasets_manifest.json" \
  --simulation-plan "papers/saish/simulation_plan.json" \
  --timeout 600
```

**Output:** 
- `results/{run_id}/error_report.json` (or `simulation_result.json`)
- `results/{run_id}/logs/stdout.txt`
- `results/{run_id}/logs/stderr.txt`
- `results/{run_id}/artifacts/*` (plots, CSV files, etc.)

---

### **Phase 4: Error Fixing (Manual)**

#### 8. **ErrorFeedbackAgent**
**Required:**
- `--error-report` (string): Path to error_report.json from SimulationRunnerAgent
- `--simulation-plan` (string): Path to simulation_plan.json

**Optional:**
- `--original-code` (string): Path to simulation.py
- `--output` (string): Path to save fix_request.json

**Example:**
```bash
python agents/ErrorFeedbackAgent.py \
  --error-report "results/{run_id}/error_report.json" \
  --simulation-plan "papers/saish/simulation_plan.json" \
  --original-code "papers/saish/simulation.py" \
  --output "results/{run_id}/fix_request.json"
```

**Output:** `results/{run_id}/fix_request.json`

**Note:** After this, merge fix_request with simulation_plan and regenerate code:
```bash
TEMP_PLAN=$(python3 -c "import json; import tempfile; f=open('results/{run_id}/fix_request.json'); d=json.load(f); p=d['simulation_plan'].copy(); p.update({'_fix_instructions':d['fix_instructions'],'_error_context':d['error_context'],'_error_summary':d['error_summary'],'_explanation':d['explanation']}); t=tempfile.NamedTemporaryFile(mode='w',suffix='.json',delete=False); json.dump(p,t,indent=2); t.close(); print(t.name)")
python agents/CodeGeneratorAgent.py --plan "$TEMP_PLAN" --datasets "papers/saish/datasets_manifest.json"
rm "$TEMP_PLAN"
```

---

### **Phase 5: Report Generation**

#### 9. **ReportAgent**
**Required:**
- `--paper-understanding` (string): Path to understand.json
- `--knowledge-graph` (string): Path to knowledge_graph.json
- `--hypothesis` (string): Path to hypothesis.json
- `--simulation-plan` (string): Path to simulation_plan.json
- `--simulation-result` (string): Path to error_report.json or simulation_result.json

**Optional:**
- `--error-history` (string): Path to JSON list of error reports
- `--output` (string): Path to save report.json
- `--markdown-output` (string): Path to save report.md

**Example:**
```bash
python agents/ReportAgent.py \
  --paper-understanding "papers/saish/understand.json" \
  --knowledge-graph "papers/saish/knowledge_graph.json" \
  --hypothesis "papers/saish/hypothesis.json" \
  --simulation-plan "papers/saish/simulation_plan.json" \
  --simulation-result "results/{run_id}/error_report.json" \
  --output "papers/saish/report.json" \
  --markdown-output "papers/saish/report.md"
```

**Output:**
- `papers/{paper_name}/report.json`
- `papers/{paper_name}/report.md`

---

## Quick Reference: Required Inputs Summary

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

---

## File Flow Diagram

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
  ↓
ErrorFeedbackAgent → fix_request.json → CodeGeneratorAgent (regenerate)
  ↓
ReportAgent → report.json, report.md
```

---

## Common Issues & Solutions

1. **Missing datasets_manifest.json**: Run DatasetAgent before CodeGeneratorAgent
2. **Wrong file paths**: All paths are relative to project root
3. **Dimension mismatch errors**: Use ErrorFeedbackAgent → regenerate code
4. **File not found errors**: Ensure dataset paths are relative to script directory
5. **Report generation fails**: Ensure simulation completed successfully first

