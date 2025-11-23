#!/usr/bin/env python3
"""
CodeGeneratorAgent - SpoonOS Agent

Converts a simulation plan JSON into a complete, runnable Python script.
Intelligently infers required libraries and generates modular, executable code.

Uses OpenAI GPT-4o (best available for code generation) to create production-ready Python scripts.
"""

import sys
import json
import argparse
import asyncio
import re
from pathlib import Path
from typing import Dict, Any

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os
from openai import AsyncOpenAI


async def generate_python_code(simulation_plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a complete runnable Python script from a simulation plan.
    
    Args:
        simulation_plan: Dictionary from SimulationPlanAgent with equations, constants, variables, procedure, outcomes
    
    Returns:
        Dictionary with python_code field containing the full script
    """
    # Prepare input for OpenAI
    input_text = f"""You are a Code Generator Agent. Your job is to convert a simulation plan into a complete, runnable Python script.

SIMULATION PLAN:
{json.dumps(simulation_plan, indent=2)}

Your responsibilities:

1. INFER REQUIRED LIBRARIES by analyzing the simulation plan:
   - If equations involve arrays, math, or numeric computation → use numpy
   - If differential equations appear → use scipy.integrate
   - If probability distributions appear → use numpy/scipy.stats
   - If graph or network terms appear → use networkx
   - If ML training loops appear → use sklearn or torch (only if explicitly required)
   - If plotting is mentioned → use matplotlib
   - If nothing specific is indicated → default to numpy + matplotlib

2. GENERATE A SINGLE COMPLETE PYTHON SCRIPT that includes:
   - All necessary imports at the top
   - Initialization of ALL constants from constants_required
   - Functions that correspond directly to blocks of the plan:
     * simulation_equations implementation (as Python functions)
     * variable sweeps (loops over variables_to_vary)
     * helper computations
     * data collection and storage
     * metrics/statistics calculations
   - A procedure that follows procedure_steps EXACTLY
   - Any required loops over temperature, time steps, networks, iterations, alpha parameters, etc.
   - Clear plotting functions if the plan expects visual output
   - A main() function that executes the full simulation
   - if __name__ == "__main__": main() block

3. THE SCRIPT MUST:
   - Be fully runnable as a standalone Python file
   - Have NO placeholders, TODOs, or pseudocode
   - NOT hallucinate details beyond what's in the simulation plan
   - Be clean, modular, and readable with clear function names
   - Validate dimensions/inputs where needed
   - Handle edge cases gracefully
   - Include comments explaining key steps
   - Use appropriate data structures (arrays, lists, dictionaries as needed)

4. DOMAIN AGNOSTIC:
   - Must work for ANY scientific domain: biology, chemistry, physics, network science, 
     computer science, ML, algorithms, epidemiology, optimization, etc.
   - Adapt the code style to the domain (e.g., network algorithms use networkx, 
     differential equations use scipy.integrate, etc.)

5. CODE QUALITY:
   - Use descriptive variable names
   - Break complex operations into functions
   - Add docstrings for main functions
   - Use numpy arrays for numerical computations
   - Use matplotlib for all plotting
   - Make the code production-ready (no debugging prints unless necessary)

Return ONLY valid JSON in this exact format:
{{
  "python_code": "#!/usr/bin/env python3\\n\\nimport numpy as np\\nimport matplotlib.pyplot as plt\\n# ... full complete Python script here ...\\n\\nif __name__ == '__main__':\\n    main()"
}}

The python_code field must contain the ENTIRE runnable Python script as a single string with proper newlines (\\n).
Return ONLY the JSON, no additional text or explanation."""

    # Use OpenAI API
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise Exception("OPENAI_API_KEY not found in environment variables. Please set it in .env file.")
    
    client = AsyncOpenAI(api_key=api_key)
    # Use GPT-4o for code generation (best available for code)
    # Note: GPT-5.1 doesn't exist yet, using gpt-4o which is the best available
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a Code Generator Agent that converts simulation plans into complete, runnable Python scripts. You are an expert in scientific computing, numerical methods, and Python best practices. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": input_text
                }
            ],
            temperature=0.2,  # Low temperature for precise code generation
            max_tokens=4096  # Maximum tokens for gpt-4-turbo (gpt-4o supports more but we use safe limit)
        )
        
        llm_response = response.choices[0].message.content
        
    except Exception as e:
        # Try fallback to gpt-4-turbo if gpt-4o fails
        if "model" in str(e).lower() or "not found" in str(e).lower():
            print(f"Warning: {model} not available, trying gpt-4-turbo", file=sys.stderr)
            try:
                model = "gpt-4-turbo"
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a Code Generator Agent that converts simulation plans into complete, runnable Python scripts. You are an expert in scientific computing, numerical methods, and Python best practices. Always respond with valid JSON only."
                        },
                        {
                            "role": "user",
                            "content": input_text
                        }
                    ],
                    temperature=0.2,
                    max_tokens=4096  # Maximum for gpt-4-turbo
                )
                llm_response = response.choices[0].message.content
            except Exception as e2:
                raise Exception(f"OpenAI API call failed with both models. Last error: {e2}")
        else:
            raise Exception(f"OpenAI API call failed: {e}")
    
    # Parse JSON response
    try:
        # Try to extract JSON from response (in case there's extra text)
        json_match = re.search(r'\{[\s\S]*\}', llm_response)
        if json_match:
            result = json.loads(json_match.group(0))
        else:
            result = json.loads(llm_response)
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse JSON response: {e}\nResponse was: {llm_response[:500]}")
    
    # Validate response structure
    if "python_code" not in result:
        raise Exception("LLM response missing required field: python_code")
    
    if not isinstance(result["python_code"], str):
        raise Exception("python_code field must be a string")
    
    # Validate that the code looks like Python
    python_code = result["python_code"]
    if not python_code.strip():
        raise Exception("python_code field is empty")
    
    # Basic validation: should contain some Python keywords
    python_keywords = ["import", "def", "if", "for", "return"]
    if not any(keyword in python_code for keyword in python_keywords):
        raise Exception("Generated code doesn't appear to be valid Python")
    
    return result


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
    """Main function to run the CodeGeneratorAgent"""
    parser = argparse.ArgumentParser(
        description="Generates a complete runnable Python script from a simulation plan"
    )
    parser.add_argument(
        "--plan",
        type=str,
        required=True,
        help="Path to JSON file from SimulationPlanAgent (simulation plan with equations, constants, variables, procedure, outcomes)"
    )
    
    args = parser.parse_args()
    
    # Load simulation plan
    try:
        simulation_plan = load_json_from_file(args.plan)
    except Exception as e:
        print(f"Error loading simulation plan file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Validate simulation plan structure
    required_fields = ["simulation_equations", "constants_required", "variables_to_vary", "procedure_steps", "expected_outcomes"]
    for field in required_fields:
        if field not in simulation_plan:
            print(f"Error: Simulation plan missing required field: {field}", file=sys.stderr)
            sys.exit(1)
    
    try:
        print("Generating Python code from simulation plan...", file=sys.stderr)
        
        # Generate Python code
        result = asyncio.run(generate_python_code(simulation_plan))
        
        # Determine output directory (same as simulation plan file directory)
        plan_path = Path(args.plan)
        output_dir = plan_path.parent
        
        # Save Python code to simulation.py in the same directory
        python_code = result["python_code"]
        output_path = output_dir / "simulation.py"
        
        with open(output_path, 'w') as f:
            f.write(python_code)
        
        print(f"Python code saved to: {output_path}", file=sys.stderr)
        
        # Also output JSON with the code
        print(json.dumps(result, indent=2))
        
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

