#!/bin/bash
# Complete pipeline for saish paper

set -e  # Exit on error

PAPER_DIR="papers/saish"
PDF_FILE="$PAPER_DIR/saish paper.pdf"

echo "=========================================="
echo "Running complete pipeline for saish paper"
echo "=========================================="

# Step 1: Paper Understanding
echo ""
echo "Step 1: Paper Understanding..."
python agents/PaperUnderstandingAgent.py --pdf "$PDF_FILE"

# Step 2: Knowledge Graph
echo ""
echo "Step 2: Building Knowledge Graph..."
python agents/KnowledgeGraphAgent.py --input "$PAPER_DIR/understand.json"

# Step 3: Hypothesis Generation
echo ""
echo "Step 3: Generating Hypothesis..."
python agents/HypothesisAgent.py --input "$PAPER_DIR/understand.json" --kg "$PAPER_DIR/knowledge_graph.json"

# Step 4: Simulation Plan
echo ""
echo "Step 4: Creating Simulation Plan..."
python agents/SimulationPlanAgent.py --hypothesis "$PAPER_DIR/hypothesis.json" --paper "$PAPER_DIR/understand.json" --kg "$PAPER_DIR/knowledge_graph.json"

# Step 5: Code Generation
echo ""
echo "Step 5: Generating Simulation Code..."
python agents/CodeGeneratorAgent.py --plan "$PAPER_DIR/simulation_plan.json"

# Step 6: Dataset Generation
echo ""
echo "Step 6: Generating Datasets..."
python agents/DatasetAgent.py --plan "$PAPER_DIR/simulation_plan.json"

# Step 7: Run Simulation
echo ""
echo "Step 7: Running Simulation..."
python agents/SimulationRunnerAgent.py \
  --python-file "$PAPER_DIR/simulation.py" \
  --datasets "$PAPER_DIR/datasets_manifest.json" \
  --simulation-plan "$PAPER_DIR/simulation_plan.json" \
  --auto-fix \
  --timeout 600

# Step 8: Generate Report
echo ""
echo "Step 8: Generating Report..."
LATEST_RESULT=$(ls -td results/*/error_report.json 2>/dev/null | head -1 | xargs dirname)
if [ -z "$LATEST_RESULT" ]; then
    echo "Error: No simulation result found. Check if simulation completed successfully."
    exit 1
fi

python agents/ReportAgent.py \
  --paper-understanding "$PAPER_DIR/understand.json" \
  --knowledge-graph "$PAPER_DIR/knowledge_graph.json" \
  --hypothesis "$PAPER_DIR/hypothesis.json" \
  --simulation-plan "$PAPER_DIR/simulation_plan.json" \
  --simulation-result "$LATEST_RESULT/error_report.json" \
  --output "$PAPER_DIR/report.json" \
  --markdown-output "$PAPER_DIR/report.md"

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "Report saved to: $PAPER_DIR/report.md"
echo "=========================================="

