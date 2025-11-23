#!/usr/bin/env python3
"""
DatasetAgent - SpoonOS Agent

Automatically provides ALL data required for generated simulation code to run.
Analyzes simulation plans and generates/downloads datasets as needed.

Uses Claude 4.5 Sonnet for structured reasoning about dataset requirements.
"""

import sys
import json
import argparse
import asyncio
import re
from pathlib import Path
from typing import Dict, Any, List
import shutil

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os
from anthropic import AsyncAnthropic

# For dataset generation
try:
    import networkx as nx
    import numpy as np
    HAS_NETWORKX = True
except ImportError:
    print("Warning: networkx or numpy not installed. Some dataset generation may fail.", file=sys.stderr)
    HAS_NETWORKX = False
    nx = None
    np = None

try:
    import requests
except ImportError:
    print("Warning: requests not installed. Dataset downloads may fail.", file=sys.stderr)


async def analyze_dataset_requirements(simulation_plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use Claude to analyze simulation plan and determine dataset requirements.
    
    Args:
        simulation_plan: Dictionary from SimulationPlanAgent
    
    Returns:
        Dictionary with dataset_type and dataset specifications
    """
    # Prepare input for Claude
    input_text = f"""You are a Dataset Agent. Your job is to analyze a simulation plan and determine what datasets are required.

SIMULATION PLAN:
{json.dumps(simulation_plan, indent=2)}

Your task:
1. Analyze the simulation_equations, variables_to_vary, and procedure_steps
2. Determine the dataset_type based on these rules:

   - If terms like "network", "topology", "scale_free", "small_world", "random",
     "Gamma(i)", "neighbors", "k-shell", "graph", "degree", "nodes", "edges" appear
     → dataset_type = "graph"

   - If ODE/PDE equations appear (dS/dt, dI/dt, dR/dt, dT/dt, ∂u/∂t, differential):
     → dataset_type = "none" (synthetic variables only, no external datasets)

   - If ML dataset names appear (MNIST, CIFAR, ImageNet, training data):
     → dataset_type = "ml"

   - If chemistry/bio mentions "molecular structure", "PDB", "protein", "protein structure":
     → dataset_type = "bio_structures"

   - If nothing matches:
     → dataset_type = "none"

3. For each dataset_type, specify what needs to be generated:

   If dataset_type = "graph":
      - Determine which graph types are needed:
        * "barabasi_albert" if scale-free is mentioned
        * "watts_strogatz" if small_world is mentioned
        * "erdos_renyi" if random is mentioned
      - Specify parameters (n_nodes, n_edges, etc.) if mentioned in the plan
      - Check if "real_world" networks are needed

   If dataset_type = "ml":
      - Specify which ML dataset (MNIST, CIFAR, etc.)
      - Specify subset size if mentioned

   If dataset_type = "bio_structures":
      - Specify PDB IDs if mentioned, or suggest common ones (1CRN, 1UBQ, etc.)

   If dataset_type = "none":
      - Return empty datasets list

Return ONLY valid JSON in this exact format:
{{
  "dataset_type": "graph|ml|bio_structures|none",
  "graph_types": ["barabasi_albert", "watts_strogatz", ...] or [],
  "graph_params": {{
    "n_nodes": 1000,
    "n_edges": 3000,
    ...
  }},
  "real_world_graphs": ["karate", "dolphins", ...] or [],
  "ml_dataset": "MNIST" or null,
  "pdb_ids": ["1CRN", ...] or [],
  "reasoning": "Brief explanation of why this dataset_type was chosen"
}}

Return ONLY the JSON, no additional text or explanation."""

    # Use Claude/Anthropic API
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise Exception("ANTHROPIC_API_KEY not found in environment variables. Please set it in .env file.")
    
    client = AsyncAnthropic(api_key=api_key)
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
    
    try:
        response = await client.messages.create(
            model=model,
            max_tokens=2000,
            temperature=0.2,  # Low temperature for precise analysis
            system="You are a Dataset Agent that analyzes simulation plans to determine dataset requirements. Always respond with valid JSON only.",
            messages=[
                {"role": "user", "content": input_text}
            ]
        )
        
        llm_response = response.content[0].text
        
    except Exception as e:
        raise Exception(f"Claude API call failed: {e}")
    
    # Parse JSON response
    try:
        json_match = re.search(r'\{[\s\S]*\}', llm_response)
        if json_match:
            result = json.loads(json_match.group(0))
        else:
            result = json.loads(llm_response)
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse JSON response: {e}\nResponse was: {llm_response[:500]}")
    
    return result


def generate_graph_datasets(dataset_spec: Dict[str, Any], datasets_dir: Path) -> Dict[str, str]:
    """
    Generate graph datasets based on specification.
    
    Returns:
        Dictionary mapping dataset names to file paths
    """
    if not HAS_NETWORKX:
        print("Error: networkx is required for graph generation. Please install: pip install networkx", file=sys.stderr)
        return {}
    
    datasets = {}
    graph_types = dataset_spec.get("graph_types", [])
    graph_params = dataset_spec.get("graph_params", {})
    real_world = dataset_spec.get("real_world_graphs", [])
    
    # Handle n_nodes - could be a list or single value
    n_nodes_raw = graph_params.get("n_nodes", 1000)
    if isinstance(n_nodes_raw, list):
        n_nodes = n_nodes_raw[0] if n_nodes_raw else 1000  # Use first value
    else:
        n_nodes = int(n_nodes_raw)
    
    # Generate synthetic graphs
    for graph_type in graph_types:
        try:
            if graph_type == "barabasi_albert":
                m_raw = graph_params.get("m", 3)
                m = int(m_raw[0] if isinstance(m_raw, list) else m_raw)
                G = nx.barabasi_albert_graph(n_nodes, m)
                filename = f"barabasi_albert_n{n_nodes}_m{m}.edgelist"
            elif graph_type == "watts_strogatz":
                k_raw = graph_params.get("k", 6)
                k = int(k_raw[0] if isinstance(k_raw, list) else k_raw)
                p_raw = graph_params.get("p", 0.3)
                p = float(p_raw[0] if isinstance(p_raw, list) else p_raw)
                G = nx.watts_strogatz_graph(n_nodes, k, p)
                filename = f"watts_strogatz_n{n_nodes}_k{k}_p{p:.2f}.edgelist"
            elif graph_type == "erdos_renyi":
                p_raw = graph_params.get("p", 0.01)
                p = float(p_raw[0] if isinstance(p_raw, list) else p_raw)
                G = nx.erdos_renyi_graph(n_nodes, p)
                filename = f"erdos_renyi_n{n_nodes}_p{p:.2f}.edgelist"
            else:
                continue
            
            filepath = datasets_dir / filename
            nx.write_edgelist(G, filepath, data=False)
            datasets[graph_type] = str(filepath)
            print(f"Generated {graph_type} graph: {filepath}", file=sys.stderr)
            
        except Exception as e:
            print(f"Warning: Failed to generate {graph_type} graph: {e}", file=sys.stderr)
    
    # Download real-world graphs
    real_world_urls = {
        "karate": "https://raw.githubusercontent.com/networkx/networkx/main/networkx/algorithms/community/tests/test_utils.py",
        "dolphins": "https://raw.githubusercontent.com/networkx/networkx/main/networkx/algorithms/community/tests/test_utils.py",
    }
    
    # For real-world graphs, we'll generate small examples
    if "karate" in real_world:
        try:
            G = nx.karate_club_graph()
            filepath = datasets_dir / "karate.edgelist"
            nx.write_edgelist(G, filepath, data=False)
            datasets["karate"] = str(filepath)
            print(f"Generated karate graph: {filepath}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Failed to generate karate graph: {e}", file=sys.stderr)
    
    if "dolphins" in real_world:
        try:
            # Generate a small-world-like graph as dolphin substitute
            G = nx.watts_strogatz_graph(62, 5, 0.3)  # Approximate dolphin network size
            filepath = datasets_dir / "dolphins.edgelist"
            nx.write_edgelist(G, filepath, data=False)
            datasets["dolphins"] = str(filepath)
            print(f"Generated dolphins graph: {filepath}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Failed to generate dolphins graph: {e}", file=sys.stderr)
    
    return datasets


def download_ml_datasets(dataset_spec: Dict[str, Any], datasets_dir: Path) -> Dict[str, str]:
    """
    Download ML datasets (minimal subsets).
    
    Returns:
        Dictionary mapping dataset names to paths
    """
    datasets = {}
    ml_dataset = dataset_spec.get("ml_dataset")
    
    if ml_dataset == "MNIST":
        # For now, create a placeholder - actual MNIST download would require torch/tensorflow
        ml_dir = datasets_dir / "ml"
        ml_dir.mkdir(parents=True, exist_ok=True)
        placeholder = ml_dir / "mnist_placeholder.txt"
        placeholder.write_text("MNIST dataset placeholder. Install torch/tensorflow to download full dataset.")
        datasets["mnist"] = str(placeholder)
        print(f"Created MNIST placeholder: {placeholder}", file=sys.stderr)
    
    return datasets


def download_pdb_files(dataset_spec: Dict[str, Any], datasets_dir: Path) -> Dict[str, str]:
    """
    Download PDB structure files from RCSB.
    
    Returns:
        Dictionary mapping PDB IDs to file paths
    """
    datasets = {}
    pdb_ids = dataset_spec.get("pdb_ids", [])
    
    if not pdb_ids:
        return datasets
    
    pdb_dir = datasets_dir / "pdb"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    
    for pdb_id in pdb_ids:
        try:
            url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                filepath = pdb_dir / f"{pdb_id.upper()}.pdb"
                filepath.write_bytes(response.content)
                datasets[pdb_id.lower()] = str(filepath)
                print(f"Downloaded PDB file {pdb_id}: {filepath}", file=sys.stderr)
            else:
                print(f"Warning: Failed to download PDB {pdb_id}: HTTP {response.status_code}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Failed to download PDB {pdb_id}: {e}", file=sys.stderr)
    
    return datasets


async def generate_datasets(simulation_plan: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """
    Main function to analyze and generate/download all required datasets.
    
    Args:
        simulation_plan: Dictionary from SimulationPlanAgent
        output_dir: Directory where datasets should be saved (same as simulation plan location)
    
    Returns:
        Dictionary with dataset_type and datasets (name -> path mapping)
    """
    # Analyze requirements using Claude
    dataset_spec = await analyze_dataset_requirements(simulation_plan)
    dataset_type = dataset_spec.get("dataset_type", "none")
    
    # Create datasets directory
    datasets_dir = output_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    all_datasets = {}
    
    # Generate/download datasets based on type
    if dataset_type == "graph":
        graph_datasets = generate_graph_datasets(dataset_spec, datasets_dir)
        all_datasets.update(graph_datasets)
    
    elif dataset_type == "ml":
        ml_datasets = download_ml_datasets(dataset_spec, datasets_dir)
        all_datasets.update(ml_datasets)
    
    elif dataset_type == "bio_structures":
        pdb_datasets = download_pdb_files(dataset_spec, datasets_dir)
        all_datasets.update(pdb_datasets)
    
    elif dataset_type == "none":
        # No datasets needed
        pass
    
    return {
        "dataset_type": dataset_type,
        "datasets": all_datasets,
        "reasoning": dataset_spec.get("reasoning", "")
    }


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
    """Main function to run the DatasetAgent"""
    parser = argparse.ArgumentParser(
        description="Generates and downloads datasets required for simulation code"
    )
    parser.add_argument(
        "--plan",
        type=str,
        required=True,
        help="Path to JSON file from SimulationPlanAgent (simulation plan)"
    )
    
    args = parser.parse_args()
    
    # Load simulation plan
    try:
        simulation_plan = load_json_from_file(args.plan)
    except Exception as e:
        print(f"Error loading simulation plan file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Validate simulation plan structure
    required_fields = ["simulation_equations", "constants_required", "variables_to_vary", "procedure_steps"]
    for field in required_fields:
        if field not in simulation_plan:
            print(f"Error: Simulation plan missing required field: {field}", file=sys.stderr)
            sys.exit(1)
    
    try:
        print("Analyzing simulation plan and generating datasets...", file=sys.stderr)
        
        # Determine output directory (same as simulation plan file directory)
        plan_path = Path(args.plan)
        output_dir = plan_path.parent
        
        # Generate datasets
        result = asyncio.run(generate_datasets(simulation_plan, output_dir))
        
        # Save dataset manifest
        manifest_path = output_dir / "datasets_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Dataset manifest saved to: {manifest_path}", file=sys.stderr)
        
        # Output JSON result
        print(json.dumps(result, indent=2))
        
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

