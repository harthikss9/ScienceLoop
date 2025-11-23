#!/bin/bash
# Manual error fixing - step by step commands

PAPER_DIR="papers/saish"
SIMULATION_FILE="$PAPER_DIR/simulation.py"
DATASETS_MANIFEST="$PAPER_DIR/datasets_manifest.json"
SIMULATION_PLAN="$PAPER_DIR/simulation_plan.json"

echo "=========================================="
echo "Manual Error Fixing Process"
echo "=========================================="
echo ""
echo "Step 1: Run simulation and check for errors"
echo "Command:"
echo "python agents/SimulationRunnerAgent.py --python-file \"$SIMULATION_FILE\" --datasets \"$DATASETS_MANIFEST\" --simulation-plan \"$SIMULATION_PLAN\" --timeout 600"
echo ""
echo "Step 2: If error found, get latest error report:"
echo "LATEST_RESULT=\$(ls -td results/*/error_report.json 2>/dev/null | head -1)"
echo "echo \"Error report: \$LATEST_RESULT\""
echo ""
echo "Step 3: Generate fix request:"
echo "python agents/ErrorFeedbackAgent.py --error-report \"\$LATEST_RESULT\" --simulation-plan \"$SIMULATION_PLAN\" --original-code \"$SIMULATION_FILE\" --output \"\$(dirname \$LATEST_RESULT)/fix_request.json\""
echo ""
echo "Step 4: Regenerate code with fixes (use Python to merge):"
echo "python3 -c \"
import json
import tempfile
with open('\$(dirname \$LATEST_RESULT)/fix_request.json', 'r') as f:
    fix_request = json.load(f)
plan_with_fixes = fix_request['simulation_plan'].copy()
plan_with_fixes['_fix_instructions'] = fix_request['fix_instructions']
plan_with_fixes['_error_context'] = fix_request['error_context']
plan_with_fixes['_error_summary'] = fix_request['error_summary']
plan_with_fixes['_explanation'] = fix_request['explanation']
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
    json.dump(plan_with_fixes, tmp, indent=2)
    print(tmp.name)
\""
echo ""
echo "Step 5: Use the temp file path with CodeGeneratorAgent:"
echo "python agents/CodeGeneratorAgent.py --plan \"<temp_file_path>\" --datasets \"$DATASETS_MANIFEST\""
echo ""
echo "Step 6: Repeat from Step 1 until no errors"
echo ""
echo "Step 7: Once successful, generate report:"
echo "LATEST_RESULT=\$(ls -td results/*/error_report.json 2>/dev/null | head -1)"
echo "python agents/ReportAgent.py --paper-understanding \"$PAPER_DIR/understand.json\" --knowledge-graph \"$PAPER_DIR/knowledge_graph.json\" --hypothesis \"$PAPER_DIR/hypothesis.json\" --simulation-plan \"$SIMULATION_PLAN\" --simulation-result \"\$LATEST_RESULT\" --output \"$PAPER_DIR/report.json\" --markdown-output \"$PAPER_DIR/report.md\""
echo ""

