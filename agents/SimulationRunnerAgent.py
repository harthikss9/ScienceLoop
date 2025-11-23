#!/usr/bin/env python3
"""
SimulationRunnerAgent - SpoonOS Agent

Executes generated Python simulation files, captures outputs, and stores
all artifacts in structured directories.

Uses pure Python (subprocess, os, shutil) - NO MCP tools.
"""

import sys
import json
import argparse
import asyncio
import uuid
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os


def validate_inputs(
    python_file_path: str,
    dataset_paths: Optional[Union[Dict[str, str], List[str]]] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validate that all required files exist.
    
    Args:
        python_file_path: Path to the Python simulation file
        dataset_paths: Dictionary or list of dataset file paths
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Validate Python simulation file
    python_path = Path(python_file_path)
    if not python_path.exists():
        return False, f"Simulation Python file not found: {python_file_path}"
    
    if not python_path.is_file():
        return False, f"Path is not a file: {python_file_path}"
    
    # Validate datasets if provided
    if dataset_paths:
        if isinstance(dataset_paths, dict):
            paths_to_check = list(dataset_paths.values())
        elif isinstance(dataset_paths, list):
            paths_to_check = dataset_paths
        else:
            return False, f"dataset_paths must be dict or list, got {type(dataset_paths)}"
        
        for dataset_path in paths_to_check:
            dataset_file = Path(dataset_path)
            if not dataset_file.exists():
                return False, f"Required dataset file not found: {dataset_path}"
    
    return True, None


def create_run_directory(base_dir: str = "results") -> Path:
    """
    Create a unique run directory structure.
    
    Args:
        base_dir: Base directory for results (default: "results")
        
    Returns:
        Path to the created run directory
    """
    run_id = str(uuid.uuid4())
    results_base = Path(base_dir)
    run_dir = results_base / run_id
    logs_dir = run_dir / "logs"
    artifacts_dir = run_dir / "artifacts"
    
    # Create directories
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    return run_dir


def run_simulation(
    python_file_path: str,
    work_dir: Optional[str] = None,
    timeout: int = 600
) -> Dict[str, Any]:
    """
    Execute the Python simulation using subprocess.
    
    Args:
        python_file_path: Path to the Python simulation file
        work_dir: Working directory for the simulation (default: parent of python_file_path)
        timeout: Timeout in seconds (default: 600 = 10 minutes)
        
    Returns:
        Dictionary with stdout, stderr, and exit_code
    """
    abs_python_path = Path(python_file_path).resolve()
    
    if work_dir is None:
        work_dir = str(abs_python_path.parent)
    else:
        work_dir = str(Path(work_dir).resolve())
    
    try:
        # Run the simulation
        result = subprocess.run(
            ["python3", str(abs_python_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=work_dir
        )
        
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"Simulation timed out after {timeout} seconds",
            "exit_code": -1
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": f"Execution error: {e}",
            "exit_code": -1
        }


def collect_artifacts(
    simulation_work_dir: str,
    artifacts_dir: Path,
    exclude_patterns: Optional[List[str]] = None
) -> List[Dict[str, str]]:
    """
    Collect all generated output files from the simulation directory.
    
    Args:
        simulation_work_dir: Directory where simulation was run
        artifacts_dir: Directory to copy artifacts to
        exclude_patterns: List of filename patterns to exclude
        
    Returns:
        List of dictionaries with filename and path for each artifact
    """
    if exclude_patterns is None:
        exclude_patterns = [
            'understand.json',
            'knowledge_graph.json',
            'hypothesis.json',
            'simulation_plan.json',
            'datasets_manifest.json',
            'simulation.py',
            '.pdf'  # Exclude original PDFs
        ]
    
    output_extensions = ['.png', '.jpg', '.jpeg', '.csv', '.txt', '.json', '.svg']
    artifacts = []
    
    work_path = Path(simulation_work_dir)
    
    if not work_path.exists():
        return artifacts
    
    # Scan directory recursively
    for file_path in work_path.rglob('*'):
        if not file_path.is_file():
            continue
        
        filename = file_path.name
        
        # Skip if matches exclude patterns
        if any(pattern in filename for pattern in exclude_patterns):
            continue
        
        # Check if it's an output file type
        if any(filename.lower().endswith(ext) for ext in output_extensions):
            try:
                # Copy to artifacts directory
                artifact_path = artifacts_dir / filename
                
                # Handle potential filename conflicts
                counter = 1
                while artifact_path.exists():
                    stem = artifact_path.stem
                    suffix = artifact_path.suffix
                    artifact_path = artifacts_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                # Copy file
                shutil.copy2(file_path, artifact_path)
                
                artifacts.append({
                    "filename": artifact_path.name,
                    "path": str(artifact_path)
                })
            except Exception as e:
                print(f"Warning: Failed to copy artifact {filename}: {e}", file=sys.stderr)
    
    return artifacts


def generate_error_summary(stderr: str, exit_code: int) -> str:
    """
    Generate a short, readable error summary from stderr and exit code.
    
    Args:
        stderr: Standard error output
        exit_code: Exit code from simulation
        
    Returns:
        Short error summary string
    """
    if exit_code == -1:
        if "timeout" in stderr.lower():
            return "Simulation timed out"
        return "Execution failed with unknown error"
    
    if exit_code != 0:
        # Try to extract key error message
        lines = stderr.strip().split('\n')
        for line in reversed(lines):
            if line.strip() and not line.startswith(' '):
                if len(line) < 200:
                    return line[:200]
                else:
                    return line[:197] + "..."
    
    return "Unknown error occurred"


async def run_simulation_agent(
    python_file_path: str,
    dataset_paths: Optional[Union[Dict[str, str], List[str]]] = None,
    simulation_workdir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main agent function to run simulation and collect results.
    
    Args:
        python_file_path: Path to the Python simulation file
        dataset_paths: Dictionary or list of dataset file paths
        simulation_workdir: Working directory for simulation (default: parent of python_file_path)
        
    Returns:
        Dictionary with status, run_id, stdout, stderr, exit_code, artifacts, results_path
    """
    # Validate inputs
    is_valid, error_msg = validate_inputs(python_file_path, dataset_paths)
    if not is_valid:
        # Return error payload
        run_dir = create_run_directory()
        run_id = run_dir.name
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Save error to log
        (logs_dir / "error.txt").write_text(error_msg)
        
        return {
            "status": "error",
            "run_id": run_id,
            "stdout": "",
            "stderr": error_msg,
            "exit_code": -1,
            "error_summary": error_msg,
            "results_path": str(run_dir)
        }
    
    # Create run directory
    run_dir = create_run_directory()
    run_id = run_dir.name
    logs_dir = run_dir / "logs"
    artifacts_dir = run_dir / "artifacts"
    
    # Determine simulation working directory
    abs_python_path = Path(python_file_path).resolve()
    if simulation_workdir is None:
        sim_work_dir = str(abs_python_path.parent)
    else:
        sim_work_dir = str(Path(simulation_workdir).resolve())
    
    # Run simulation
    print(f"Executing simulation: {python_file_path}", file=sys.stderr)
    if dataset_paths:
        count = len(dataset_paths) if isinstance(dataset_paths, dict) else len(dataset_paths)
        print(f"Using {count} dataset(s)", file=sys.stderr)
    
    result = run_simulation(python_file_path, work_dir=sim_work_dir)
    
    stdout = result["stdout"]
    stderr = result["stderr"]
    exit_code = result["exit_code"]
    
    # Ensure logs directory exists
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Save logs
    (logs_dir / "stdout.txt").write_text(stdout)
    (logs_dir / "stderr.txt").write_text(stderr)
    
    # Collect artifacts
    artifacts = collect_artifacts(sim_work_dir, artifacts_dir)
    
    if artifacts:
        print(f"Captured {len(artifacts)} artifact(s)", file=sys.stderr)
    
    # Return structured result based on success/failure
    if exit_code == 0:
        success_result = {
            "status": "success",
            "run_id": run_id,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "artifacts": artifacts,
            "results_path": str(run_dir)
        }
        
        # Save result report for ReportAgent (always save, success or error)
        result_report_path = run_dir / "simulation_result.json"
        with open(result_report_path, 'w') as f:
            json.dump(success_result, f, indent=2)
        
        # Also save as error_report.json for backward compatibility (ReportAgent can use either)
        error_report_path = run_dir / "error_report.json"
        with open(error_report_path, 'w') as f:
            json.dump(success_result, f, indent=2)
        
        print(f"Result report saved to: {result_report_path}", file=sys.stderr)
        
        return success_result
    else:
        error_summary = generate_error_summary(stderr, exit_code)
        error_result = {
            "status": "error",
            "run_id": run_id,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "error_summary": error_summary,
            "artifacts": artifacts,
            "results_path": str(run_dir)
        }
        
        # Save error report for ErrorFeedbackAgent and ReportAgent
        error_report_path = run_dir / "error_report.json"
        with open(error_report_path, 'w') as f:
            json.dump(error_result, f, indent=2)
        
        # Also save as simulation_result.json for consistency
        result_report_path = run_dir / "simulation_result.json"
        with open(result_report_path, 'w') as f:
            json.dump(error_result, f, indent=2)
        
        print(f"Error report saved to: {error_report_path}", file=sys.stderr)
        print("Run ErrorFeedbackAgent to generate fix request:", file=sys.stderr)
        print(f"  python agents/ErrorFeedbackAgent.py --error-report {error_report_path} --simulation-plan <plan.json>", file=sys.stderr)
        
        return error_result


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
        description="SimulationRunnerAgent - Execute Python simulations and collect results"
    )
    parser.add_argument(
        "--python-file",
        type=str,
        required=True,
        help="Path to the Python simulation file"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        help="Path to JSON file containing dataset paths (dict or list)"
    )
    parser.add_argument(
        "--workdir",
        type=str,
        help="Working directory for simulation (default: parent of python_file)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds (default: 600)"
    )
    parser.add_argument(
        "--auto-fix",
        action="store_true",
        help="Automatically trigger ErrorFeedbackAgent on errors (requires --simulation-plan)"
    )
    parser.add_argument(
        "--simulation-plan",
        type=str,
        help="Path to simulation plan JSON (required for --auto-fix)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of auto-fix retry attempts (default: 3)"
    )
    
    args = parser.parse_args()
    
    # Load dataset paths if provided
    dataset_paths = None
    if args.datasets:
        try:
            datasets_data = load_json_from_file(args.datasets)
            if isinstance(datasets_data, dict):
                # If it's a dict with "datasets" key, extract it
                if "datasets" in datasets_data:
                    dataset_paths = datasets_data["datasets"]
                else:
                    dataset_paths = datasets_data
            elif isinstance(datasets_data, list):
                dataset_paths = datasets_data
            else:
                print(f"Warning: datasets file format not recognized, ignoring", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Failed to load datasets: {e}", file=sys.stderr)
    
    try:
        max_retries = args.max_retries if args.auto_fix else 0
        retry_count = 0
        result = None
        
        # Retry loop: keep fixing and re-running until success or max retries
        while retry_count <= max_retries:
            # Run the simulation agent
            result = asyncio.run(run_simulation_agent(
                python_file_path=args.python_file,
                dataset_paths=dataset_paths,
                simulation_workdir=args.workdir
            ))
            
            # If successful, break out of retry loop
            if result.get("status") == "success":
                print(f"\n✓ Simulation completed successfully!", file=sys.stderr)
                break
            
            # If error occurred and auto-fix is enabled, try to fix
            if result.get("status") == "error" and args.auto_fix and retry_count < max_retries:
                if not args.simulation_plan:
                    print("Error: --simulation-plan required when using --auto-fix", file=sys.stderr)
                    sys.exit(1)
                
                print("\n" + "="*60, file=sys.stderr)
                print(f"Auto-triggering ErrorFeedbackAgent (attempt {retry_count + 1}/{max_retries})...", file=sys.stderr)
                print("="*60 + "\n", file=sys.stderr)
                
                # Import and call ErrorFeedbackAgent
                try:
                    # Add parent directory to path for imports (sys is already imported at top)
                    parent_dir = str(Path(__file__).parent.parent)
                    if parent_dir not in sys.path:
                        sys.path.insert(0, parent_dir)
                    from agents.ErrorFeedbackAgent import generate_fix_request
                
                    error_report_path = Path(result["results_path"]) / "error_report.json"
                    if not error_report_path.exists():
                        # Create error report if it doesn't exist
                        with open(error_report_path, 'w') as f:
                            json.dump(result, f, indent=2)
                    
                    # Load simulation plan using local function
                    simulation_plan = load_json_from_file(args.simulation_plan)
                    
                    # Try to load original code
                    original_code = None
                    code_path = Path(args.python_file)
                    if code_path.exists():
                        original_code = code_path.read_text()
                    
                    # Generate fix request
                    fix_request = generate_fix_request(
                        error_report=result,
                        simulation_plan=simulation_plan,
                        original_python_code=original_code
                    )
                    
                    # Save fix request
                    fix_request_path = Path(result["results_path"]) / "fix_request.json"
                    with open(fix_request_path, 'w') as f:
                        json.dump(fix_request, f, indent=2)
                    
                    print(f"Fix request saved to: {fix_request_path}", file=sys.stderr)
                    
                    # Automatically trigger CodeGeneratorAgent to regenerate code
                    print("\n" + "="*60, file=sys.stderr)
                    print("Auto-triggering CodeGeneratorAgent to regenerate code...", file=sys.stderr)
                    print("="*60 + "\n", file=sys.stderr)
                    
                    try:
                        from agents.CodeGeneratorAgent import generate_python_code
                        # asyncio is already imported at the top
                        
                        # Extract simulation plan from fix request
                        plan_with_fixes = fix_request["simulation_plan"].copy()
                        # Add fix instructions to the plan so CodeGeneratorAgent can use them
                        plan_with_fixes["_fix_instructions"] = fix_request["fix_instructions"]
                        plan_with_fixes["_error_context"] = fix_request["error_context"]
                        plan_with_fixes["_error_summary"] = fix_request["error_summary"]
                        plan_with_fixes["_explanation"] = fix_request["explanation"]
                        
                        # Load datasets manifest if available
                        datasets_manifest = None
                        if args.datasets:
                            try:
                                datasets_manifest = load_json_from_file(args.datasets)
                            except Exception as e:
                                print(f"Warning: Failed to load datasets manifest for code regeneration: {e}", file=sys.stderr)
                        
                        # Generate corrected code with datasets manifest
                        code_result = asyncio.run(generate_python_code(plan_with_fixes, datasets_manifest))
                        
                        # Save regenerated code
                        code_path = Path(args.python_file)
                        output_path = code_path.parent / "simulation.py"
                        with open(output_path, 'w') as f:
                            f.write(code_result["python_code"])
                        
                        print(f"Regenerated code saved to: {output_path}", file=sys.stderr)
                        print(f"\nRetrying simulation (attempt {retry_count + 2}/{max_retries + 1})...", file=sys.stderr)
                        retry_count += 1
                        # Continue loop to re-run with fixed code
                        continue
                        
                    except Exception as e:
                        print(f"Warning: Failed to auto-regenerate code: {e}", file=sys.stderr)
                        import traceback
                        traceback.print_exc()
                        print("\nFix request (manual regeneration required):", file=sys.stderr)
                        print(json.dumps(fix_request, indent=2))
                        # Break out of retry loop if code regeneration fails
                        break
                
                except Exception as e:
                    print(f"Warning: Failed to auto-trigger ErrorFeedbackAgent: {e}", file=sys.stderr)
                    print("Error result:", file=sys.stderr)
                    print(json.dumps(result, indent=2))
                    # Break out of retry loop if ErrorFeedbackAgent fails
                    break
            else:
                # If auto-fix is disabled or max retries reached, break
                break
        
        # Output final result (success or final error after all retries)
        if result:
            if result.get("status") == "error" and retry_count >= max_retries:
                print(f"\n✗ Simulation failed after {max_retries + 1} attempts. Final error:", file=sys.stderr)
            print(json.dumps(result, indent=2))
        else:
            print("Error: No simulation result available", file=sys.stderr)
            sys.exit(1)
        
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

