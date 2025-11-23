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
import ast
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os
from openai import AsyncOpenAI


async def generate_python_code(simulation_plan: Dict[str, Any], datasets_manifest: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generate a complete runnable Python script from a simulation plan.
    
    Args:
        simulation_plan: Dictionary from SimulationPlanAgent with equations, constants, variables, procedure, outcomes
        datasets_manifest: Optional dictionary from DatasetAgent with dataset_type and datasets paths
    
    Returns:
        Dictionary with python_code field containing the full script
    """
    # Check if this is a fix request (has fix instructions)
    has_fixes = "_fix_instructions" in simulation_plan
    fix_context = ""
    if has_fixes:
        fix_instructions = simulation_plan.pop("_fix_instructions", [])
        error_context = simulation_plan.pop("_error_context", "")
        error_summary = simulation_plan.pop("_error_summary", "")
        explanation = simulation_plan.pop("_explanation", "")
        
        fix_context = f"""

IMPORTANT: This is a CODE REGENERATION REQUEST with fixes needed.

PREVIOUS ERROR:
{error_summary}

ERROR CONTEXT:
{error_context}

WHY IT FAILED:
{explanation}

FIX INSTRUCTIONS (MUST APPLY THESE - READ CAREFULLY):
{chr(10).join(f"- {instruction}" for instruction in fix_instructions)}

CRITICAL REQUIREMENTS:
- You MUST regenerate the code with ALL these fixes applied
- Do NOT repeat the same errors - if you see the same error again, you failed
- Pay special attention to file path fixes - use paths relative to script directory
- If fix instructions mention changing a path, CHANGE IT EXACTLY as specified
- For dimension mismatch errors: CAREFULLY analyze your loop structure and ensure x and y arrays have matching lengths
- If fix instructions say "move append outside inner loops" - DO IT. Check where you're appending values.
- If fix instructions mention nested loops causing dimension issues - restructure your loops accordingly
- Verify the fix addresses the root cause before generating code
- READ EACH FIX INSTRUCTION CAREFULLY and apply it explicitly in your code
"""
    
    # Prepare dataset information for the prompt
    dataset_info = ""
    if datasets_manifest:
        # Convert dataset paths to be relative to script directory
        # Script is always saved as simulation.py in papers/{paper_name}/, so working dir is papers/{paper_name}/
        converted_datasets = {}
        for name, path in datasets_manifest.get("datasets", {}).items():
            # Extract relative path from papers/{paper_name}/datasets/... to datasets/...
            path_obj = Path(path)
            # Find the papers/{paper_name} part and get everything after it
            parts = path_obj.parts
            if "papers" in parts:
                papers_idx = parts.index("papers")
                if papers_idx + 1 < len(parts):
                    # Get everything after papers/{paper_name}/
                    relative_parts = parts[papers_idx + 2:]
                    converted_path = str(Path(*relative_parts))
                    converted_datasets[name] = converted_path
                else:
                    converted_datasets[name] = path
            else:
                converted_datasets[name] = path
        
        dataset_info = f"""

AVAILABLE DATASETS (from DatasetAgent):
Original paths: {json.dumps(datasets_manifest.get("datasets", {}), indent=2)}
Converted paths (relative to script directory): {json.dumps(converted_datasets, indent=2)}

CRITICAL: You MUST use the CONVERTED paths (relative to script directory) in your generated code.
- The simulation runs from the directory containing simulation.py (e.g., papers/saish/)
- Use the converted paths shown above (e.g., "datasets/synthetic/classification_data.csv")
- Use pandas.read_csv() for CSV files, numpy.load() for .npy files, networkx.read_*() for graph files
- Example: if converted path is "datasets/synthetic/classification_data.csv", use:
  data = pd.read_csv('datasets/synthetic/classification_data.csv')
- ALWAYS use paths relative to the script's directory, NOT absolute paths
- Do NOT generate synthetic data inline if datasets are already provided - use the provided files
"""
    
    # Prepare input for OpenAI
    input_text = f"""You are a Code Generator Agent. Your job is to convert a simulation plan into a complete, runnable Python script.
{fix_context}
SIMULATION PLAN:
{json.dumps(simulation_plan, indent=2)}
{dataset_info}

CRITICAL PLOTTING INSTRUCTIONS:
- ONLY import matplotlib if the simulation plan explicitly mentions plotting, visualization, graphs, or charts
- If plotting is needed, add at the very top (BEFORE other imports):
  import matplotlib
  matplotlib.use('Agg')  # Non-interactive backend to prevent hanging
  import matplotlib.pyplot as plt
- If NO plotting is mentioned in the plan, DO NOT import matplotlib at all
- Check expected_outcomes and procedure_steps for plot-related terms before deciding

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
   - CRITICAL LOOP STRUCTURE RULES:
     * If you have nested loops and plan to plot results, ensure you collect ONE value per x-axis point
     * Example CORRECT: for x_val in x_range: acc = compute(); results.append(acc)  # One per x
     * Example WRONG: for x_val in x_range: for y_val in y_range: acc = compute(); results.append(acc)  # Multiple per x
     * If plotting against outer loop variable, only append inside outer loop, NOT in inner loops
     * Before plotting, verify: len(x_array) == len(y_array)
   - INTELLIGENT PLOTTING DECISION:
     * Only generate plots if the simulation plan explicitly mentions visualization, plotting, graphs, charts, or visual output
     * Check expected_outcomes and procedure_steps for plot-related keywords: "plot", "graph", "chart", "visualize", "figure", "diagram"
     * If NO plotting keywords are found, DO NOT generate any plotting code (no matplotlib imports for plotting, no plt.savefig calls)
     * If plotting IS mentioned, then:
       - ALWAYS save plots with plt.savefig("plot_name.png", dpi=300, bbox_inches='tight')
       - NEVER use plt.show() - it blocks execution
       - Use plt.close() after saving each plot
       - CRITICAL: Before plt.plot(), verify x and y arrays have same length: assert len(x_array) == len(y_array), "Dimension mismatch: x and y must have same length"
     * For data-only outputs (CSV, JSON, text), save files without plots
   - A main() function that executes the full simulation
   - if __name__ == "__main__": main() block

3. THE SCRIPT MUST:
   - Be fully runnable as a standalone Python file
   - Have NO placeholders, TODOs, or pseudocode - EVERY function must be fully implemented
   - NOT hallucinate details beyond what's in the simulation plan
   - Be clean, modular, and readable with clear function names
   - Validate dimensions/inputs where needed (especially for sklearn compatibility)
   - Handle edge cases gracefully
   - Include comments explaining key steps
   - Use appropriate data structures (arrays, lists, dictionaries as needed)
   - CRITICAL: If using sklearn, ensure all data passed to models is numpy arrays of numeric types
   - CRITICAL: Convert any dictionary-based data structures to numpy arrays before sklearn operations
   - CRITICAL: Implement ALL steps from procedure_steps - do not leave any step as a placeholder

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
   - IMPORTANT: Always save plots using plt.savefig() with descriptive filenames
   - NEVER use plt.show() - it blocks execution and opens windows
   - After saving plots, use plt.close() to free memory
   - Make the code production-ready (no debugging prints unless necessary)

6. CRITICAL DATA STRUCTURE RULES:
   - When generating data for sklearn models, ALWAYS return numpy arrays of numbers, NOT dictionaries
   - Example CORRECT: np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2D array of floats
   - Example WRONG: np.array([dict(feature1=1.0), dict(feature1=2.0)])  # Array of dicts - WILL FAIL
   - If you create dictionaries, convert them to numpy arrays BEFORE returning:
     * CORRECT: features = np.array([[row['f1'], row['f2']] for row in data_list])
     * WRONG: features = np.array(data_list)  # This creates array of dicts
   - For sklearn: X must be 2D numpy array (n_samples, n_features) of numeric types
   - For sklearn: y must be 1D numpy array (n_samples,) of numeric types
   - ALWAYS verify data shapes: print("X shape:", X.shape, "y shape:", y.shape) before model training
   - If generating synthetic data, build feature matrix directly as list of lists, then convert to np.array

7. COMMON MISTAKES TO AVOID:
   - DO NOT return np.array(list_of_dicts) - sklearn cannot handle arrays of dictionaries
   - DO NOT forget to convert dictionaries to numeric arrays before sklearn operations
   - DO NOT use placeholders or incomplete implementations - every function must be fully implemented
   - DO NOT skip error handling for edge cases (empty arrays, division by zero, etc.)
   - DO NOT forget to implement ALL steps from procedure_steps - every step must have real code
   - DO NOT create incomplete main() functions - all planned steps must execute
   - DO NOT forget to save outputs (CSV files, plots) - use explicit file paths
   - DO NOT use plt.show() - always use plt.savefig() + plt.close()
   - DO NOT collect multiple values per x-axis point in nested loops when plotting
   - CRITICAL: If plotting, ensure collection loop matches x-axis loop exactly
   - Example WRONG: for x_val in range(11): for y_val in range(5): results.append(compute()) then plt.plot(range(11), results)  # 55 values vs 11 x-values
   - Example CORRECT: for x_val in range(11): acc = compute(); results.append(acc) then plt.plot(range(11), results)  # 11 values vs 11 x-values

8. VALIDATION AND TESTING:
   - After generating data, verify shapes and types match sklearn requirements
   - Before model training, check: assert X.ndim == 2, "X must be 2D array"
   - Before model training, check: assert y.ndim == 1, "y must be 1D array"
   - After model training, verify predictions are generated
   - Add print statements showing progress: "Step X: [description] completed"
   - Verify all expected outputs are generated (files saved, plots created)
   - BEFORE PLOTTING: Always add assert len(x_array) == len(y_array), "Dimension mismatch: x and y arrays must have same length"
   - If collecting results in nested loops, verify collection logic matches plotting requirements

9. CRITICAL LOOP AND PLOTTING PATTERN:
   CORRECT PATTERN for plotting:
   ```python
   results = []
   x_values = range(11)  # or np.arange(11)
   for x_val in x_values:
       # Do computation for this x value
       result = compute_something(x_val)
       results.append(result)  # ONE append per x value
   
   # Verify before plotting
   assert len(x_values) == len(results), "Mismatch: x-values and results must have same length"
   plt.plot(x_values, results)
   ```
   
   WRONG PATTERN (will cause dimension error):
   ```python
   results = []
   x_values = range(11)
   for x_val in x_values:
       for y_val in y_range:  # Nested loop!
           result = compute(x_val, y_val)
           results.append(result)  # Multiple appends per x - WRONG!
   
   plt.plot(x_values, results)  # ERROR: 11 x-values vs many results
   ```
   
   If you need nested loops but want to plot against outer loop:
   - Collect ONE aggregated value per outer loop iteration
   - Example: for x_val in x_range: all_results = []; for y_val in y_range: all_results.append(compute(x_val,y_val)); results.append(np.mean(all_results))

10. EXAMPLE: CORRECT DATA GENERATION FOR SKLEARN:
   ```python
   def generate_data(n_samples):
       X = []  # List of feature vectors
       y = []  # List of labels
       
       for i in range(n_samples):
           # Create feature vector as a list of numbers
           features = [
               np.random.normal(165, 25),  # feature 1
               np.random.normal(250, 30),  # feature 2
               np.random.normal(100, 20),   # feature 3
           ]
           X.append(features)  # Append list, not dict
           y.append(1 if i < n_samples * 0.75 else 0)
       
       # Convert to numpy arrays
       X = np.array(X)  # Shape: (n_samples, n_features)
       y = np.array(y)  # Shape: (n_samples,)
       
       return X, y
   ```
   
   WRONG EXAMPLE (DO NOT DO THIS):
   ```python
   def generate_data(n_samples):
       data = []
       for i in range(n_samples):
           data.append(dict(f1=1.0, f2=2.0))  # Dictionary
       return np.array(data), labels  # WRONG! Creates array of dicts
   ```

BEFORE RETURNING CODE, REVIEW IT FOR THESE CRITICAL ISSUES:
1. Check loop structure: If plotting, ensure collection loop matches x-axis loop (same number of iterations)
2. Verify dimensions: Count how many times you append to results/accuracies vs x-axis length
3. Check file paths: All dataset paths must be relative to script directory (not project root)
4. Verify data structures: No arrays of dictionaries for sklearn
5. Check plotting: If plotting, verify len(x_array) == len(y_array) before plt.plot()

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
    
    # Validate Python syntax
    try:
        ast.parse(python_code)
    except SyntaxError as e:
        raise Exception(f"Generated code has syntax errors: {e}\nLine {e.lineno}: {e.text}")
    
    # Check for common issues
    if "np.array([{" in python_code or "np.array([dict" in python_code:
        print("WARNING: Code may contain numpy arrays of dictionaries - this will fail with sklearn!", file=sys.stderr)
    
    # Check for placeholders
    placeholder_keywords = ["TODO", "FIXME", "placeholder", "not implemented", "pass  #"]
    found_placeholders = [kw for kw in placeholder_keywords if kw.lower() in python_code.lower()]
    if found_placeholders:
        print(f"WARNING: Code contains placeholder keywords: {found_placeholders}", file=sys.stderr)
    
    # Check for plt.show() usage
    if "plt.show()" in python_code:
        print("WARNING: Code contains plt.show() - this will block execution!", file=sys.stderr)
    
    # Validate plotting dimension consistency
    if "plt.plot" in python_code:
        lines = python_code.split('\n')
        plot_line_idx = None
        for i, line in enumerate(lines):
            if "plt.plot" in line and not line.strip().startswith("#"):
                plot_line_idx = i
                break
        
        if plot_line_idx is not None:
            before_plot = '\n'.join(lines[:plot_line_idx])
            
            # Extract x and y variables from plt.plot line
            plot_line = lines[plot_line_idx]
            plot_match = re.search(r'plt\.plot\s*\(\s*([^,]+)\s*,\s*([^,)]+)', plot_line)
            
            if plot_match:
                x_var = plot_match.group(1).strip()
                y_var = plot_match.group(2).strip()
                
                # Check for nested loops collecting y_var
                loop_depth = 0
                max_loop_depth = 0
                append_in_nested = False
                
                for line in lines[:plot_line_idx]:
                    stripped = line.strip()
                    if stripped.startswith("for ") and not stripped.startswith("#"):
                        loop_depth += 1
                        max_loop_depth = max(max_loop_depth, loop_depth)
                    elif stripped.startswith(("if ", "elif ", "else:", "def ", "class ")):
                        pass  # Don't count these
                    elif stripped.endswith(":") and loop_depth > 0:
                        pass  # Don't reset on colons
                    elif any(keyword in stripped for keyword in ["break", "continue", "return"]):
                        if loop_depth > 0:
                            loop_depth -= 1
                    elif f"{y_var}.append" in stripped or ".append(" in stripped:
                        if loop_depth > 1:
                            append_in_nested = True
                    if stripped and not stripped.startswith("#") and not stripped.endswith(":"):
                        # Check for dedent (simple heuristic)
                        if len(line) - len(line.lstrip()) == 0 and loop_depth > 0:
                            loop_depth = 0
                
                if max_loop_depth > 1 and append_in_nested:
                    print("CRITICAL WARNING: Detected dimension mismatch risk!", file=sys.stderr)
                    print(f"WARNING: Found nested loops (depth {max_loop_depth}) with append operations", file=sys.stderr)
                    print(f"WARNING: Plotting {x_var} vs {y_var} may fail if dimensions don't match", file=sys.stderr)
                    print("WARNING: Code may have nested loops collecting multiple values per x-axis point", file=sys.stderr)
                    print("WARNING: If this code fails with dimension error, ensure append() is only called once per outer loop iteration", file=sys.stderr)
    
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
    parser.add_argument(
        "--datasets",
        type=str,
        help="Path to JSON file from DatasetAgent (datasets manifest with dataset_type and datasets paths) - optional"
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
    
    # Load datasets manifest if provided
    datasets_manifest = None
    if args.datasets:
        try:
            datasets_manifest = load_json_from_file(args.datasets)
            print(f"Loaded datasets manifest: {len(datasets_manifest.get('datasets', {}))} dataset(s)", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Failed to load datasets manifest: {e}", file=sys.stderr)
            print("Continuing without dataset information...", file=sys.stderr)
    
    try:
        print("Generating Python code from simulation plan...", file=sys.stderr)
        
        # Generate Python code
        result = asyncio.run(generate_python_code(simulation_plan, datasets_manifest))
        
        # Determine output directory
        # If datasets manifest is provided, use its directory (papers/{paper_name}/)
        # Otherwise, use simulation plan file directory
        if datasets_manifest and "datasets" in datasets_manifest:
            # Extract paper directory from first dataset path
            first_dataset_path = list(datasets_manifest["datasets"].values())[0]
            dataset_path = Path(first_dataset_path)
            # Find papers/{paper_name}/ in the path
            parts = dataset_path.parts
            if "papers" in parts:
                papers_idx = parts.index("papers")
                if papers_idx + 1 < len(parts):
                    paper_name = parts[papers_idx + 1]
                    output_dir = Path("papers") / paper_name
                else:
                    output_dir = Path(args.plan).parent
            else:
                output_dir = Path(args.plan).parent
        else:
            # Fallback to simulation plan file directory
            plan_path = Path(args.plan)
            output_dir = plan_path.parent
        
        # Save Python code to simulation.py in the determined directory
        python_code = result["python_code"]
        output_path = output_dir / "simulation.py"
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
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

