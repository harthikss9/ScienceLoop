#!/bin/bash
# Manual error fixing loop - run until no errors

set -e

PAPER_DIR="papers/saish"
SIMULATION_FILE="$PAPER_DIR/simulation.py"
DATASETS_MANIFEST="$PAPER_DIR/datasets_manifest.json"
SIMULATION_PLAN="$PAPER_DIR/simulation_plan.json"
MAX_ITERATIONS=10

echo "=========================================="
echo "Manual Error Fixing Loop"
echo "=========================================="
echo ""

iteration=0

while [ $iteration -lt $MAX_ITERATIONS ]; do
    iteration=$((iteration + 1))
    echo "--- Iteration $iteration ---"
    
    # Run simulation
    echo "Running simulation..."
    python agents/SimulationRunnerAgent.py \
        --python-file "$SIMULATION_FILE" \
        --datasets "$DATASETS_MANIFEST" \
        --simulation-plan "$SIMULATION_PLAN" \
        --timeout 600 > /tmp/simulation_output.json 2>&1
    
    # Extract result
    RESULT_JSON=$(cat /tmp/simulation_output.json | grep -A 1000 '^{' | head -100)
    
    # Check if there's an error
    if echo "$RESULT_JSON" | grep -q '"status": "error"'; then
        echo "❌ Error detected!"
        
        # Find the latest error report
        LATEST_RESULT=$(ls -td results/*/error_report.json 2>/dev/null | head -1)
        if [ -z "$LATEST_RESULT" ]; then
            echo "Error: No error report found"
            exit 1
        fi
        
        RESULT_DIR=$(dirname "$LATEST_RESULT")
        echo "Error report: $LATEST_RESULT"
        
        # Generate fix request
        echo "Generating fix request..."
        python agents/ErrorFeedbackAgent.py \
            --error-report "$LATEST_RESULT" \
            --simulation-plan "$SIMULATION_PLAN" \
            --original-code "$SIMULATION_FILE" \
            --output "$RESULT_DIR/fix_request.json"
        
        # Regenerate code with fix
        echo "Regenerating code with fixes..."
        # Use Python to properly merge fix request with simulation plan
        python3 << PYTHON_SCRIPT
import json
import sys

# Load fix request
with open("$RESULT_DIR/fix_request.json", 'r') as f:
    fix_request = json.load(f)

# Merge simulation plan with fix instructions
plan_with_fixes = fix_request["simulation_plan"].copy()
plan_with_fixes["_fix_instructions"] = fix_request["fix_instructions"]
plan_with_fixes["_error_context"] = fix_request["error_context"]
plan_with_fixes["_error_summary"] = fix_request["error_summary"]
plan_with_fixes["_explanation"] = fix_request["explanation"]

# Save temporary plan
import tempfile
import os
temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
json.dump(plan_with_fixes, temp_file, indent=2)
temp_file.close()

print(temp_file.name)
PYTHON_SCRIPT
        
        PLAN_WITH_FIXES=$(python3 << PYTHON_SCRIPT
import json
import tempfile
import os

with open("$RESULT_DIR/fix_request.json", 'r') as f:
    fix_request = json.load(f)

plan_with_fixes = fix_request["simulation_plan"].copy()
plan_with_fixes["_fix_instructions"] = fix_request["fix_instructions"]
plan_with_fixes["_error_context"] = fix_request["error_context"]
plan_with_fixes["_error_summary"] = fix_request["error_summary"]
plan_with_fixes["_explanation"] = fix_request["explanation"]

temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
json.dump(plan_with_fixes, temp_file, indent=2)
temp_file.close()
print(temp_file.name)
PYTHON_SCRIPT
)
        
        python agents/CodeGeneratorAgent.py \
            --plan "$PLAN_WITH_FIXES" \
            --datasets "$DATASETS_MANIFEST"
        
        rm "$PLAN_WITH_FIXES"
        
        echo "Code regenerated. Will retry..."
        echo ""
        sleep 2
    else
        echo "✅ Simulation completed successfully!"
        echo ""
        echo "Latest result directory: $(ls -td results/* 2>/dev/null | head -1)"
        break
    fi
done

if [ $iteration -ge $MAX_ITERATIONS ]; then
    echo "⚠️  Reached maximum iterations ($MAX_ITERATIONS). Stopping."
    exit 1
fi

echo "=========================================="
echo "All errors fixed! Ready for report generation."
echo "=========================================="

