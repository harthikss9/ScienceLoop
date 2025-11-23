# Complete Pipeline for MLDL Paper

## Complete Pipeline - Copy-Paste Ready:

### **PHASE 1: Paper Analysis & Planning (Run Once)**

```bash
python agents/PaperUnderstandingAgent.py --pdf "papers/mldl/1-s2.0-S0957417410009097-main.pdf"
python agents/KnowledgeGraphAgent.py --input "papers/mldl/understand.json"
python agents/HypothesisAgent.py --input "papers/mldl/understand.json" --kg "papers/mldl/knowledge_graph.json"
python agents/SimulationPlanAgent.py --hypothesis "papers/mldl/hypothesis.json" --paper "papers/mldl/understand.json" --kg "papers/mldl/knowledge_graph.json"
```

### **PHASE 2: Code & Data Preparation (Run Once)**

```bash
python agents/DatasetAgent.py --plan "papers/mldl/simulation_plan.json"
python agents/CodeGeneratorAgent.py --plan "papers/mldl/simulation_plan.json" --datasets "papers/mldl/datasets_manifest.json"
```

### **PHASE 3: Manual Error Fixing Loop (Repeat Until Success)**

**3a. Run Simulation:**

```bash
python agents/SimulationRunnerAgent.py --python-file "papers/mldl/simulation.py" --datasets "papers/mldl/datasets_manifest.json" --simulation-plan "papers/mldl/simulation_plan.json" --timeout 600
```

**3b. If Error Found - Fix Loop:**

```bash
LATEST_RESULT=$(ls -td results/*/error_report.json 2>/dev/null | head -1)
python agents/ErrorFeedbackAgent.py --error-report "$LATEST_RESULT" --simulation-plan "papers/mldl/simulation_plan.json" --original-code "papers/mldl/simulation.py" --output "$(dirname $LATEST_RESULT)/fix_request.json"
TEMP_PLAN=$(python3 -c "import json; import tempfile; f=open('$(dirname $LATEST_RESULT)/fix_request.json'); d=json.load(f); p=d['simulation_plan'].copy(); p.update({'_fix_instructions':d['fix_instructions'],'_error_context':d['error_context'],'_error_summary':d['error_summary'],'_explanation':d['explanation']}); t=tempfile.NamedTemporaryFile(mode='w',suffix='.json',delete=False); json.dump(p,t,indent=2); t.close(); print(t.name)")
python agents/CodeGeneratorAgent.py --plan "$TEMP_PLAN" --datasets "papers/mldl/datasets_manifest.json"
rm "$TEMP_PLAN"
```

**Then repeat Step 3a until status is "success".**

### **PHASE 4: Report Generation (Only After Success)**

```bash
LATEST_RESULT=$(ls -td results/*/simulation_result.json 2>/dev/null | head -1)
python agents/ReportAgent.py --paper-understanding "papers/mldl/understand.json" --knowledge-graph "papers/mldl/knowledge_graph.json" --hypothesis "papers/mldl/hypothesis.json" --simulation-plan "papers/mldl/simulation_plan.json" --simulation-result "$LATEST_RESULT" --output "papers/mldl/report.json" --markdown-output "papers/mldl/report.md"
```

---

## Flow Summary:

- **Phase 1-2**: Run once sequentially
- **Phase 3**: Run simulation → check for errors → fix if needed → repeat until success
- **Phase 4**: Only run after simulation succeeds

Full guide saved in `RUN_MLDL_PAPER.md`.

---

## Expected Output Files

After completing all phases, you should have:

```
papers/mldl/
├── 1-s2.0-S0957417410009097-main.pdf
├── understand.json
├── knowledge_graph.json
├── hypothesis.json
├── simulation_plan.json
├── datasets_manifest.json
├── datasets/
│   └── ...
├── simulation.py
├── report.json
└── report.md
```
