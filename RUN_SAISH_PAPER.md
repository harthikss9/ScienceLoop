# Complete Pipeline for Saish Paper

## Full Flow from Start to Finish

### Phase 1: Paper Analysis & Planning (Steps 1-4)

**Step 1: Paper Understanding**
```bash
python agents/PaperUnderstandingAgent.py --pdf "papers/saish/saish paper.pdf"
```

**Step 2: Knowledge Graph**
```bash
python agents/KnowledgeGraphAgent.py --input "papers/saish/understand.json"
```

**Step 3: Hypothesis Generation**
```bash
python agents/HypothesisAgent.py --input "papers/saish/understand.json" --kg "papers/saish/knowledge_graph.json"
```

**Step 4: Simulation Plan**
```bash
python agents/SimulationPlanAgent.py --hypothesis "papers/saish/hypothesis.json" --paper "papers/saish/understand.json" --kg "papers/saish/knowledge_graph.json"
```

### Phase 2: Code & Data Preparation (Steps 5-6)

**Step 5: Dataset Generation**
```bash
python agents/DatasetAgent.py --plan "papers/saish/simulation_plan.json"
```

**Step 6: Code Generation (with datasets)**
```bash
python agents/CodeGeneratorAgent.py --plan "papers/saish/simulation_plan.json" --datasets "papers/saish/datasets_manifest.json"
```

### Phase 3: Manual Error Fixing Loop (Step 7)

**Step 7a: Run Simulation (Check for Errors)**
```bash
python agents/SimulationRunnerAgent.py --python-file "papers/saish/simulation.py" --datasets "papers/saish/datasets_manifest.json" --simulation-plan "papers/saish/simulation_plan.json" --timeout 600
```

**Step 7b: If Error Found - Fix Loop**

Get the latest error report:
```bash
LATEST_RESULT=$(ls -td results/*/error_report.json 2>/dev/null | head -1)
echo "Error report: $LATEST_RESULT"
```

Generate fix request:
```bash
python agents/ErrorFeedbackAgent.py --error-report "$LATEST_RESULT" --simulation-plan "papers/saish/simulation_plan.json" --original-code "papers/saish/simulation.py" --output "$(dirname $LATEST_RESULT)/fix_request.json"
```

Regenerate code with fixes:
```bash
TEMP_PLAN=$(python3 -c "import json; import tempfile; f=open('$(dirname $LATEST_RESULT)/fix_request.json'); d=json.load(f); p=d['simulation_plan'].copy(); p.update({'_fix_instructions':d['fix_instructions'],'_error_context':d['error_context'],'_error_summary':d['error_summary'],'_explanation':d['explanation']}); t=tempfile.NamedTemporaryFile(mode='w',suffix='.json',delete=False); json.dump(p,t,indent=2); t.close(); print(t.name)")
python agents/CodeGeneratorAgent.py --plan "$TEMP_PLAN" --datasets "papers/saish/datasets_manifest.json"
rm "$TEMP_PLAN"
```

**Repeat Step 7a until status is "success" (no errors)**

### Phase 4: Report Generation (Step 8 - Only After Success)

**Step 8: Generate Report**
```bash
LATEST_RESULT=$(ls -td results/*/error_report.json 2>/dev/null | head -1)
python agents/ReportAgent.py --paper-understanding "papers/saish/understand.json" --knowledge-graph "papers/saish/knowledge_graph.json" --hypothesis "papers/saish/hypothesis.json" --simulation-plan "papers/saish/simulation_plan.json" --simulation-result "$LATEST_RESULT" --output "papers/saish/report.json" --markdown-output "papers/saish/report.md"
```

## Quick Reference: All Commands in Order

```bash
# Phase 1: Analysis
python agents/PaperUnderstandingAgent.py --pdf "papers/saish/saish paper.pdf"
python agents/KnowledgeGraphAgent.py --input "papers/saish/understand.json"
python agents/HypothesisAgent.py --input "papers/saish/understand.json" --kg "papers/saish/knowledge_graph.json"
python agents/SimulationPlanAgent.py --hypothesis "papers/saish/hypothesis.json" --paper "papers/saish/understand.json" --kg "papers/saish/knowledge_graph.json"

# Phase 2: Preparation
python agents/DatasetAgent.py --plan "papers/saish/simulation_plan.json"
python agents/CodeGeneratorAgent.py --plan "papers/saish/simulation_plan.json" --datasets "papers/saish/datasets_manifest.json"

# Phase 3: Error Fixing Loop (repeat until success)
python agents/SimulationRunnerAgent.py --python-file "papers/saish/simulation.py" --datasets "papers/saish/datasets_manifest.json" --simulation-plan "papers/saish/simulation_plan.json" --timeout 600
# If error: run fix loop commands from Step 7b above

# Phase 4: Report (only after success)
LATEST_RESULT=$(ls -td results/*/error_report.json 2>/dev/null | head -1)
python agents/ReportAgent.py --paper-understanding "papers/saish/understand.json" --knowledge-graph "papers/saish/knowledge_graph.json" --hypothesis "papers/saish/hypothesis.json" --simulation-plan "papers/saish/simulation_plan.json" --simulation-result "$LATEST_RESULT" --output "papers/saish/report.json" --markdown-output "papers/saish/report.md"
```

## Notes:
- **Phase 1-2**: Run once, sequentially
- **Phase 3**: Run simulation, check for errors, fix if needed, repeat until success
- **Phase 4**: Only run after simulation succeeds (status: "success")
- CodeGeneratorAgent intelligently decides when to generate plots based on simulation plan
- All generated files are saved in `papers/saish/` directory
- Final report will be in `papers/saish/report.md` and `papers/saish/report.json`

