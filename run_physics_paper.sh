#!/bin/bash
# Complete pipeline for Physics Paper

source spoon-env/bin/activate

echo "=========================================="
echo "PHYSICS PAPER - Complete Pipeline"
echo "=========================================="

# Step 1: Paper Understanding
echo ""
echo "Step 1: Paper Understanding..."
python agents/PaperUnderstandingAgent.py --pdf "papers/physics paper/1206.1238v1.pdf"

# Step 2: Knowledge Graph
echo ""
echo "Step 2: Knowledge Graph..."
python agents/KnowledgeGraphAgent.py --input "papers/physics paper/understand.json"

# Step 3: Hypothesis Generation
echo ""
echo "Step 3: Hypothesis Generation..."
python agents/HypothesisAgent.py --input "papers/physics paper/understand.json" --kg "papers/physics paper/knowledge_graph.json"

# Step 4: Simulation Plan
echo ""
echo "Step 4: Simulation Plan..."
python agents/SimulationPlanAgent.py --hypothesis "papers/physics paper/hypothesis.json" --paper "papers/physics paper/understand.json" --kg "papers/physics paper/knowledge_graph.json"

# Step 5: Code Generation
echo ""
echo "Step 5: Code Generation..."
python agents/CodeGeneratorAgent.py --plan "papers/physics paper/simulation_plan.json"

# Step 6: Dataset Generation
echo ""
echo "Step 6: Dataset Generation..."
python agents/DatasetAgent.py --plan "papers/physics paper/simulation_plan.json"

# Step 7: Simulation Runner (with auto-fix)
echo ""
echo "Step 7: Simulation Runner..."
python agents/SimulationRunnerAgent.py \
  --python-file "papers/physics paper/simulation.py" \
  --datasets "papers/physics paper/datasets_manifest.json" \
  --simulation-plan "papers/physics paper/simulation_plan.json" \
  --auto-fix

echo ""
echo "=========================================="
echo "PHYSICS PAPER PIPELINE COMPLETE!"
echo "=========================================="

