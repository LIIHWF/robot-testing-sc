# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Batch falsification script for running STL-based falsification on all configurations.

This script reads a configuration JSON file and runs falsification on all configurations,
storing results in a specified output directory.

Example usage:
    # Run with random policy (for testing)
    python scripts/falsify_all.py --config-path data/single_all.json --output-dir ./falsify_results --use-random-policy

    # Run with inference server
    python scripts/falsify_all.py --config-path data/single_all.json --output-dir ./falsify_results --policy-port 5555

    # Run specific range of configurations
    python scripts/falsify_all.py --config-path data/single_all.json --output-dir ./falsify_results --start 1 --end 10

    # Resume from a specific configuration
    python scripts/falsify_all.py --config-path data/single_all.json --output-dir ./falsify_results --resume
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root directory to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from concrete_layer.falsifier.new_falsify import create_policy, falsify


def load_configurations(config_path: str) -> List[Dict[str, Any]]:
    """Load configurations from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_summary(summary: Dict[str, Any], output_dir: str):
    """Save the summary results to a JSON file."""
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved to {summary_path}")


def load_existing_summary(output_dir: str) -> Optional[Dict[str, Any]]:
    """Load existing summary if available for resuming."""
    summary_path = os.path.join(output_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def get_completed_configs(output_dir: str) -> set:
    """Get set of already completed configuration numbers."""
    completed = set()
    summary = load_existing_summary(output_dir)
    if summary and "results" in summary:
        for result in summary["results"]:
            if result.get("status") in ("completed", "falsified", "not_falsified"):
                completed.add(result.get("config_number"))
    return completed


def main():
    parser = argparse.ArgumentParser(
        description="Batch falsification for all configurations in a JSON file"
    )
    parser.add_argument(
        "--config-path", "-c",
        type=str,
        required=True,
        help="Path to the configurations JSON file"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        required=True,
        help="Directory to store all results"
    )
    parser.add_argument(
        "--budget", "-b",
        type=int,
        default=50,
        help="Optimization budget per configuration (default: 50)"
    )
    parser.add_argument(
        "--horizon", "-H",
        type=int,
        default=300,
        help="Rollout horizon (default: 300)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Base random seed (default: 42)"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start configuration number (1-based, inclusive)"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End configuration number (1-based, inclusive)"
    )
    parser.add_argument(
        "--policy-host",
        type=str,
        default="localhost",
        help="Inference server host (default: localhost)"
    )
    parser.add_argument(
        "--policy-port",
        type=int,
        default=5555,
        help="Inference server port (default: 5555)"
    )
    parser.add_argument(
        "--use-random-policy",
        action="store_true",
        help="Use random policy instead of inference server (for testing)"
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable video saving"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last completed configuration"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without actually running"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate config path
    if not os.path.exists(args.config_path):
        print(f"Error: Configuration file not found: {args.config_path}")
        sys.exit(1)
    
    # Load configurations
    print(f"Loading configurations from {args.config_path}...")
    configurations = load_configurations(args.config_path)
    total_configs = len(configurations)
    print(f"Found {total_configs} configurations")
    
    # Determine range
    start_idx = args.start if args.start is not None else 1
    end_idx = args.end if args.end is not None else total_configs
    
    # Validate range
    if start_idx < 1 or start_idx > total_configs:
        print(f"Error: Start index {start_idx} out of range [1, {total_configs}]")
        sys.exit(1)
    if end_idx < start_idx or end_idx > total_configs:
        print(f"Error: End index {end_idx} out of range [{start_idx}, {total_configs}]")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check for resume
    completed_configs = set()
    if args.resume:
        completed_configs = get_completed_configs(args.output_dir)
        if completed_configs:
            print(f"Resuming: {len(completed_configs)} configurations already completed")
    
    # Filter configurations to run
    configs_to_run = []
    for i in range(start_idx, end_idx + 1):
        if i not in completed_configs:
            configs_to_run.append(i)
    
    print(f"\nConfigurations to run: {len(configs_to_run)} (from {start_idx} to {end_idx})")
    
    if args.dry_run:
        print("\n[DRY RUN] Would run the following configurations:")
        for config_num in configs_to_run[:20]:  # Show first 20
            config = configurations[config_num - 1]
            task = config.get("task_expression", "N/A")
            print(f"  Config {config_num}: {task}")
        if len(configs_to_run) > 20:
            print(f"  ... and {len(configs_to_run) - 20} more")
        return
    
    # Create policy function
    if args.use_random_policy:
        print("\nUsing random policy for testing...")
        policy_fn = create_policy(use_random=True)
    else:
        try:
            print(f"\nConnecting to inference server at {args.policy_host}:{args.policy_port}...")
            policy_fn = create_policy(
                host=args.policy_host,
                port=args.policy_port,
                use_random=False
            )
            print("Connected successfully!")
        except Exception as e:
            print(f"\nError: Failed to connect to inference server: {e}")
            print("\nTo use a random policy for testing, use --use-random-policy flag.")
            sys.exit(1)
    
    # Initialize summary
    summary = {
        "config_path": os.path.abspath(args.config_path),
        "output_dir": os.path.abspath(args.output_dir),
        "total_configurations": total_configs,
        "start_index": start_idx,
        "end_index": end_idx,
        "budget": args.budget,
        "horizon": args.horizon,
        "base_seed": args.seed,
        "use_random_policy": args.use_random_policy,
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "results": [],
        "statistics": {
            "total_run": 0,
            "falsified": 0,
            "not_falsified": 0,
            "errors": 0,
        }
    }
    
    # Load existing results if resuming
    if args.resume:
        existing_summary = load_existing_summary(args.output_dir)
        if existing_summary and "results" in existing_summary:
            summary["results"] = existing_summary["results"]
            summary["statistics"] = existing_summary.get("statistics", summary["statistics"])
    
    print(f"\n{'=' * 70}")
    print(f"Starting batch falsification")
    print(f"{'=' * 70}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Budget: {args.budget}")
    print(f"  Horizon: {args.horizon}")
    print(f"  Save videos: {not args.no_video}")
    print(f"{'=' * 70}\n")
    
    start_time = time.time()
    
    for idx, config_num in enumerate(configs_to_run):
        config = configurations[config_num - 1]
        task_expression = config.get("task_expression", "N/A")
        
        print(f"\n[{idx + 1}/{len(configs_to_run)}] Configuration {config_num}: {task_expression}")
        print("-" * 70)
        
        # Create config-specific output directory
        config_output_dir = os.path.join(args.output_dir, f"config_{config_num}")
        os.makedirs(config_output_dir, exist_ok=True)
        
        # Use different seed for each configuration
        config_seed = args.seed + config_num
        
        result_entry = {
            "config_number": config_num,
            "task_expression": task_expression,
            "status": "pending",
            "falsified": None,
            "best_robustness": None,
            "error": None,
            "output_dir": config_output_dir,
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
        }
        
        try:
            result = falsify(
                config_number=config_num,
                policy=policy_fn,
                horizon=args.horizon,
                budget=args.budget,
                config_path=args.config_path,
                seed=config_seed,
                verbose=args.verbose,
                output_dir=config_output_dir,
                save_video=not args.no_video,
            )
            
            # Check if configuration was invalid
            if result.get("error"):
                result_entry["status"] = "config_invalid"
                result_entry["error"] = result["error"]
                result_entry["completed_at"] = datetime.now().isoformat()
                summary["statistics"]["errors"] += 1
                print(f"  ✗ Config invalid: {result['error']}")
            else:
                result_entry["status"] = "falsified" if result["falsified"] else "not_falsified"
                result_entry["falsified"] = result["falsified"]
                result_entry["best_robustness"] = result["best_robustness"]
                result_entry["completed_at"] = datetime.now().isoformat()
                
                summary["statistics"]["total_run"] += 1
                if result["falsified"]:
                    summary["statistics"]["falsified"] += 1
                    print(f"  ⚠️  FALSIFIED (robustness: {result['best_robustness']:.6f})")
                else:
                    summary["statistics"]["not_falsified"] += 1
                    print(f"  ✓ Not falsified (robustness: {result['best_robustness']:.6f})")
                
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Saving progress...")
            result_entry["status"] = "interrupted"
            result_entry["error"] = "Interrupted by user"
            summary["results"].append(result_entry)
            save_summary(summary, args.output_dir)
            sys.exit(1)
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"  ✗ Error: {error_msg}")
            if args.verbose:
                traceback.print_exc()
            
            result_entry["status"] = "error"
            result_entry["error"] = error_msg
            result_entry["completed_at"] = datetime.now().isoformat()
            summary["statistics"]["errors"] += 1
        
        summary["results"].append(result_entry)
        
        # Save summary after each configuration (for resume capability)
        save_summary(summary, args.output_dir)
    
    # Finalize summary
    elapsed_time = time.time() - start_time
    summary["completed_at"] = datetime.now().isoformat()
    summary["elapsed_time_seconds"] = elapsed_time
    save_summary(summary, args.output_dir)
    
    # Print final summary
    stats = summary["statistics"]
    print(f"\n{'=' * 70}")
    print(f"Batch Falsification Complete")
    print(f"{'=' * 70}")
    print(f"  Total run: {stats['total_run']}")
    print(f"  Falsified: {stats['falsified']}")
    print(f"  Not falsified: {stats['not_falsified']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Elapsed time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} minutes)")
    if stats['total_run'] > 0:
        falsification_rate = stats['falsified'] / stats['total_run'] * 100
        print(f"  Falsification rate: {falsification_rate:.1f}%")
    print(f"\nResults saved to: {args.output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()



