# Complete Pipeline for Bio Paper

## Complete Pipeline - Copy-Paste Ready:

### **PHASE 1: Paper Analysis & Planning (Run Once)**

```bash
python agents/PaperUnderstandingAgent.py --pdf "papers/bio paper/scalley97A.pdf"
python agents/KnowledgeGraphAgent.py --input "papers/bio paper/understand.json"
python agents/HypothesisAgent.py --input "papers/bio paper/understand.json" --kg "papers/bio paper/knowledge_graph.json"
python agents/SimulationPlanAgent.py --hypothesis "papers/bio paper/hypothesis.json" --paper "papers/bio paper/understand.json" --kg "papers/bio paper/knowledge_graph.json"
```

### **PHASE 2: Code & Data Preparation (Run Once)**

```bash
python agents/DatasetAgent.py --plan "papers/bio paper/simulation_plan.json"
python agents/CodeGeneratorAgent.py --plan "papers/bio paper/simulation_plan.json" --datasets "papers/bio paper/datasets_manifest.json"
```

### **PHASE 3: Manual Error Fixing Loop (Repeat Until Success)**

**3a. Run Simulation:**

```bash
python agents/SimulationRunnerAgent.py --python-file "papers/bio paper/simulation.py" --datasets "papers/bio paper/datasets_manifest.json" --simulation-plan "papers/bio paper/simulation_plan.json" --timeout 600
```

**3b. If Error Found - Fix Loop:**

```bash
LATEST_RESULT=$(ls -td results/*/error_report.json 2>/dev/null | head -1)
python agents/ErrorFeedbackAgent.py --error-report "$LATEST_RESULT" --simulation-plan "papers/bio paper/simulation_plan.json" --original-code "papers/bio paper/simulation.py" --output "$(dirname $LATEST_RESULT)/fix_request.json"
TEMP_PLAN=$(python3 -c "import json; import tempfile; f=open('$(dirname $LATEST_RESULT)/fix_request.json'); d=json.load(f); p=d['simulation_plan'].copy(); p.update({'_fix_instructions':d['fix_instructions'],'_error_context':d['error_context'],'_error_summary':d['error_summary'],'_explanation':d['explanation']}); t=tempfile.NamedTemporaryFile(mode='w',suffix='.json',delete=False); json.dump(p,t,indent=2); t.close(); print(t.name)")
python agents/CodeGeneratorAgent.py --plan "$TEMP_PLAN" --datasets "papers/bio paper/datasets_manifest.json"
rm "$TEMP_PLAN"
```

**Then repeat Step 3a until status is "success".**

### **PHASE 4: Report Generation (Only After Success)**

```bash
LATEST_RESULT=$(ls -td results/*/simulation_result.json 2>/dev/null | head -1)
python agents/ReportAgent.py --paper-understanding "papers/bio paper/understand.json" --knowledge-graph "papers/bio paper/knowledge_graph.json" --hypothesis "papers/bio paper/hypothesis.json" --simulation-plan "papers/bio paper/simulation_plan.json" --simulation-result "$LATEST_RESULT" --output "papers/bio paper/report.json" --markdown-output "papers/bio paper/report.md"
```

---

## Flow Summary:

- **Phase 1-2**: Run once sequentially
- **Phase 3**: Run simulation → check for errors → fix if needed → repeat until success
- **Phase 4**: Only run after simulation succeeds

Full guide saved in `RUN_BIO_PAPER.md`.

---

## Expected Output Files

After completing all phases, you should have:

```
papers/bio paper/
├── scalley97A.pdf
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

