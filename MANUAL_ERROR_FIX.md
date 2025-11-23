# Manual Error Fixing Process

## Step-by-Step Commands

### Step 1: Run Simulation (Check for Errors)
```bash
python agents/SimulationRunnerAgent.py --python-file "papers/saish/simulation.py" --datasets "papers/saish/datasets_manifest.json" --simulation-plan "papers/saish/simulation_plan.json" --timeout 600
```

### Step 2: If Error Found - Get Latest Error Report
```bash
LATEST_RESULT=$(ls -td results/*/error_report.json 2>/dev/null | head -1)
echo "Error report: $LATEST_RESULT"
```

### Step 3: Generate Fix Request
```bash
python agents/ErrorFeedbackAgent.py --error-report "$LATEST_RESULT" --simulation-plan "papers/saish/simulation_plan.json" --original-code "papers/saish/simulation.py" --output "$(dirname $LATEST_RESULT)/fix_request.json"
```

### Step 4: Regenerate Code with Fixes
```bash
TEMP_PLAN=$(python3 -c "
import json
import tempfile
with open('$(dirname $LATEST_RESULT)/fix_request.json', 'r') as f:
    fix_request = json.load(f)
plan_with_fixes = fix_request['simulation_plan'].copy()
plan_with_fixes['_fix_instructions'] = fix_request['fix_instructions']
plan_with_fixes['_error_context'] = fix_request['error_context']
plan_with_fixes['_error_summary'] = fix_request['error_summary']
plan_with_fixes['_explanation'] = fix_request['explanation']
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
    json.dump(plan_with_fixes, tmp, indent=2)
    print(tmp.name)
")
python agents/CodeGeneratorAgent.py --plan "$TEMP_PLAN" --datasets "papers/saish/datasets_manifest.json"
rm "$TEMP_PLAN"
```

### Step 5: Repeat Steps 1-4 Until No Errors

### Step 6: Once Successful - Generate Report
```bash
LATEST_RESULT=$(ls -td results/*/error_report.json 2>/dev/null | head -1)
python agents/ReportAgent.py --paper-understanding "papers/saish/understand.json" --knowledge-graph "papers/saish/knowledge_graph.json" --hypothesis "papers/saish/hypothesis.json" --simulation-plan "papers/saish/simulation_plan.json" --simulation-result "$LATEST_RESULT" --output "papers/saish/report.json" --markdown-output "papers/saish/report.md"
```

## Notes:
- CodeGeneratorAgent now intelligently decides when to generate plots (only if simulation plan mentions visualization/plotting)
- Keep running Step 1 until status is "success" (no errors)
- Only run Step 6 after simulation succeeds

