#!/bin/bash
# Complete pipeline commands for all papers

# Activate virtual environment
source spoon-env/bin/activate

echo "=========================================="
echo "BIO PAPER"
echo "=========================================="

# Bio Paper
python agents/PaperUnderstandingAgent.py --pdf "papers/bio paper/scalley97A.pdf"
python agents/KnowledgeGraphAgent.py --input "papers/bio paper/understand.json"
python agents/HypothesisAgent.py --input "papers/bio paper/understand.json" --kg "papers/bio paper/knowledge_graph.json"
python agents/SimulationPlanAgent.py --hypothesis "papers/bio paper/hypothesis.json" --paper "papers/bio paper/understand.json" --kg "papers/bio paper/knowledge_graph.json"
python agents/CodeGeneratorAgent.py --plan "papers/bio paper/simulation_plan.json"
python agents/DatasetAgent.py --plan "papers/bio paper/simulation_plan.json"
python agents/SimulationRunnerAgent.py --python-file "papers/bio paper/simulation.py" --datasets "papers/bio paper/datasets_manifest.json"

echo ""
echo "=========================================="
echo "PHYSICS PAPER"
echo "=========================================="

# Physics Paper
python agents/PaperUnderstandingAgent.py --pdf "papers/physics paper/1206.1238v1.pdf"
python agents/KnowledgeGraphAgent.py --input "papers/physics paper/understand.json"
python agents/HypothesisAgent.py --input "papers/physics paper/understand.json" --kg "papers/physics paper/knowledge_graph.json"
python agents/SimulationPlanAgent.py --hypothesis "papers/physics paper/hypothesis.json" --paper "papers/physics paper/understand.json" --kg "papers/physics paper/knowledge_graph.json"
python agents/CodeGeneratorAgent.py --plan "papers/physics paper/simulation_plan.json"
python agents/DatasetAgent.py --plan "papers/physics paper/simulation_plan.json"
python agents/SimulationRunnerAgent.py --python-file "papers/physics paper/simulation.py" --datasets "papers/physics paper/datasets_manifest.json"

echo ""
echo "=========================================="
echo "ML/DL PAPER"
echo "=========================================="

# ML/DL Paper
python agents/PaperUnderstandingAgent.py --pdf "papers/mldl/1-s2.0-S0957417410009097-main.pdf"
python agents/KnowledgeGraphAgent.py --input "papers/mldl/understand.json"
python agents/HypothesisAgent.py --input "papers/mldl/understand.json" --kg "papers/mldl/knowledge_graph.json"
python agents/SimulationPlanAgent.py --hypothesis "papers/mldl/hypothesis.json" --paper "papers/mldl/understand.json" --kg "papers/mldl/knowledge_graph.json"
python agents/CodeGeneratorAgent.py --plan "papers/mldl/simulation_plan.json"
python agents/DatasetAgent.py --plan "papers/mldl/simulation_plan.json"
python agents/SimulationRunnerAgent.py --python-file "papers/mldl/simulation.py" --datasets "papers/mldl/datasets_manifest.json"

echo ""
echo "=========================================="
echo "SNA PAPER"
echo "=========================================="

# SNA Paper
python agents/PaperUnderstandingAgent.py --pdf "papers/sna paper/Complexity - 2020 - Zhao - Ranking Influential Nodes in Complex Networks with Information Entropy Method.pdf"
python agents/KnowledgeGraphAgent.py --input "papers/sna paper/understand.json"
python agents/HypothesisAgent.py --input "papers/sna paper/understand.json" --kg "papers/sna paper/knowledge_graph.json"
python agents/SimulationPlanAgent.py --hypothesis "papers/sna paper/hypothesis.json" --paper "papers/sna paper/understand.json" --kg "papers/sna paper/knowledge_graph.json"
python agents/CodeGeneratorAgent.py --plan "papers/sna paper/simulation_plan.json"
python agents/DatasetAgent.py --plan "papers/sna paper/simulation_plan.json"
python agents/SimulationRunnerAgent.py --python-file "papers/sna paper/simulation.py" --datasets "papers/sna paper/datasets_manifest.json"

echo ""
echo "=========================================="
echo "ALL PAPERS COMPLETE!"
echo "=========================================="

