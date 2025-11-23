#!/usr/bin/env python3
"""
ReportAgent - SpoonOS Agent

Analyzes simulation results and generates comprehensive reports comparing
expected outcomes with actual results. Uses Claude 4.5 Sonnet for reasoning.
"""

import sys
import json
import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os
from anthropic import AsyncAnthropic


async def generate_report(
    paper_understanding: Dict[str, Any],
    knowledge_graph: Dict[str, Any],
    hypothesis: Dict[str, Any],
    simulation_plan: Dict[str, Any],
    simulation_result: Dict[str, Any],
    error_history: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Generate a comprehensive report analyzing simulation results.
    
    Args:
        paper_understanding: JSON from PaperUnderstandingAgent
        knowledge_graph: JSON from KnowledgeGraphAgent
        hypothesis: JSON from HypothesisAgent
        simulation_plan: JSON from SimulationPlanAgent
        simulation_result: JSON from SimulationRunnerAgent
        error_history: Optional list of past error reports
        
    Returns:
        Dictionary with report_markdown and summary
    """
    # Prepare input for Claude
    input_text = f"""You are a ReportAgent analyzing scientific simulation results. Your task is to generate a comprehensive, well-reasoned report comparing expected outcomes with actual results.

PAPER CONTEXT:
Paper Understanding:
{json.dumps(paper_understanding, indent=2)}

Knowledge Graph:
{json.dumps(knowledge_graph, indent=2)}

HYPOTHESIS:
{json.dumps(hypothesis, indent=2)}

SIMULATION PLAN:
Expected Outcomes: {simulation_plan.get('expected_outcomes', 'Not specified')}
Simulation Equations: {json.dumps(simulation_plan.get('simulation_equations', []), indent=2)}
Variables to Vary: {json.dumps(simulation_plan.get('variables_to_vary', []), indent=2)}
Procedure Steps: {json.dumps(simulation_plan.get('procedure_steps', []), indent=2)}

SIMULATION RESULT:
Status: {simulation_result.get('status', 'unknown')}
Exit Code: {simulation_result.get('exit_code', -1)}
Run ID: {simulation_result.get('run_id', 'unknown')}
Artifacts Generated: {json.dumps(simulation_result.get('artifacts', []), indent=2)}

STDOUT (simulation output):
{simulation_result.get('stdout', 'No output')}

STDERR (errors/warnings):
{simulation_result.get('stderr', 'No errors')}

ERROR HISTORY (if any):
{json.dumps(error_history if error_history else [], indent=2)}

Your task:

1. ANALYZE THE RESULTS:
   - If status == "error": Clearly state we did NOT reach expected outcomes and explain why based on stderr/logs
   - If status == "success": Analyze stdout and artifacts to determine if patterns described in expected_outcomes were observed
   - Look for evidence in stdout (numbers, metrics, convergence messages, etc.)
   - Reference artifacts by filename (e.g., plots, CSV files)

2. GENERATE A HUMAN-READABLE MARKDOWN REPORT with these sections:
   - Title: "Simulation Results Report"
   - Short Experiment Summary (2-3 sentences)
   - Section: "Goal & Hypothesis" (from hypothesis + paper context)
   - Section: "What We Planned to See" (from expected_outcomes)
   - Section: "What We Actually Observed"
     * Reference stdout (numbers, metrics) if present
     * Reference artifacts by filename (e.g., plots)
   - Section: "Did We Meet the Expected Outcome?"
     * Explicit YES / PARTIALLY / NO
     * A short paragraph of reasoning
   - Section: "Key Notes About the Simulation"
     * Bullet points with interesting behaviors, edge cases, limitations
   - Section: "Next Steps / Recommendations"
     * What to tweak in code, plan, or hypothesis
     * If more runs are needed, or different parameter sweeps

3. GENERATE A MACHINE-READABLE SUMMARY JSON:
{{
  "success": true | false | "partial",
  "reason": "<one-paragraph explanation>",
  "matched_expectations": [
     "<which parts of expected_outcomes were clearly observed>"
  ],
  "unmet_expectations": [
     "<which expected behaviors were not clearly observed or contradicted>"
  ],
  "key_observations": [
     "<short bullet strings for main findings>"
  ],
  "recommendations": [
     "<concrete next actions for future runs or code changes>"
  ]
}}

Return ONLY valid JSON in this exact format:
{{
  "report_markdown": "# Simulation Results Report\\n\\n[full markdown report here]",
  "summary": {{
    "success": true | false | "partial",
    "reason": "...",
    "matched_expectations": [...],
    "unmet_expectations": [...],
    "key_observations": [...],
    "recommendations": [...]
  }}
}}

Focus on strong, explicit reasoning that can be read and understood quickly. Be specific about what evidence supports or contradicts the expected outcomes."""

    # Use Anthropic Claude API
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise Exception("ANTHROPIC_API_KEY not found in environment variables. Please set it in .env file.")
    
    client = AsyncAnthropic(api_key=api_key)
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
    
    try:
        response = await client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=0.3,
            messages=[
                {
                    "role": "user",
                    "content": input_text
                }
            ]
        )
        
        # Extract text from response
        llm_response = ""
        for content_block in response.content:
            if hasattr(content_block, 'text'):
                llm_response += content_block.text
        
        # Parse JSON from response
        try:
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\{[\s\S]*\}', llm_response)
            if json_match:
                result = json.loads(json_match.group(0))
            else:
                # Try parsing the whole response
                result = json.loads(llm_response)
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse LLM response as JSON: {e}", file=sys.stderr)
            print(f"Raw response: {llm_response[:500]}", file=sys.stderr)
            # Return error structure
            return {
                "report_markdown": f"# Error Generating Report\n\nFailed to parse LLM response as JSON.\n\nRaw response:\n```\n{llm_response[:1000]}\n```",
                "summary": {
                    "success": False,
                    "reason": "Failed to parse report from LLM response",
                    "matched_expectations": [],
                    "unmet_expectations": [],
                    "key_observations": [],
                    "recommendations": []
                }
            }
        
        return result
        
    except Exception as e:
        print(f"Error calling Claude API: {e}", file=sys.stderr)
        raise


def load_json_from_file(file_path: str) -> dict:
    """Load JSON from a file path."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {file_path}: {e}")


def main():
    """Main entry point for the agent."""
    parser = argparse.ArgumentParser(
        description="ReportAgent - Generate comprehensive simulation reports"
    )
    parser.add_argument(
        "--paper-understanding",
        type=str,
        help="Path to JSON file from PaperUnderstandingAgent"
    )
    parser.add_argument(
        "--knowledge-graph",
        type=str,
        help="Path to JSON file from KnowledgeGraphAgent"
    )
    parser.add_argument(
        "--hypothesis",
        type=str,
        help="Path to JSON file from HypothesisAgent"
    )
    parser.add_argument(
        "--simulation-plan",
        type=str,
        help="Path to JSON file from SimulationPlanAgent"
    )
    parser.add_argument(
        "--simulation-result",
        type=str,
        help="Path to JSON file from SimulationRunnerAgent result"
    )
    parser.add_argument(
        "--error-history",
        type=str,
        help="Path to JSON file containing list of error reports (optional)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the report JSON (default: print to stdout)"
    )
    parser.add_argument(
        "--markdown-output",
        type=str,
        help="Path to save the markdown report (optional)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load all JSON inputs
        paper_understanding = load_json_from_file(args.paper_understanding)
        knowledge_graph = load_json_from_file(args.knowledge_graph)
        hypothesis = load_json_from_file(args.hypothesis)
        simulation_plan = load_json_from_file(args.simulation_plan)
        simulation_result = load_json_from_file(args.simulation_result)
        
        error_history = None
        if args.error_history:
            error_history = load_json_from_file(args.error_history)
        
        # Generate report
        print("Generating report...", file=sys.stderr)
        result = asyncio.run(generate_report(
            paper_understanding=paper_understanding,
            knowledge_graph=knowledge_graph,
            hypothesis=hypothesis,
            simulation_plan=simulation_plan,
            simulation_result=simulation_result,
            error_history=error_history
        ))
        
        # Save outputs
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Report JSON saved to: {output_path}", file=sys.stderr)
        
        if args.markdown_output:
            markdown_path = Path(args.markdown_output)
            markdown_path.parent.mkdir(parents=True, exist_ok=True)
            with open(markdown_path, 'w') as f:
                f.write(result["report_markdown"])
            print(f"Markdown report saved to: {markdown_path}", file=sys.stderr)
        
        # Print JSON result to stdout
        print(json.dumps(result, indent=2))
        
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

