# Validating Generalist Robots with Situation Calculus and STL Falsification

This validation framework for generalist robotic systems combines abstract reasoning with concrete system falsification. It has integrated the NVIDIA Isaac GR00T N1.5 model with the RoboCasa simulation environment.

## Project Structure

```
robot-testing-sc/
├── abstract_layer/          # Abstract test generation layer
├── concrete_layer/          # Concrete test execution layer
├── meta_model/              # Domain models and task grammar definitions
├── robot_controller/        # GR00T N1.5 robot controller integration
├── robocasa/  # RoboCasa-based simulation environment
└── scripts/                 # Main execution scripts
```

### Component Overview

- **Meta Model**: Defines the task grammar and system model using situation calculus, specifying actions, objects, and relations.

- **Abstract Layer**: Generates valid abstract test configurations for task and initial world state using combinatorial testing.

- **Concrete Layer**: Instantiates abstract configurations into concrete test cases and executes them in simulation with STL falsification.

- **Robot Controller**: Integration with NVIDIA Isaac GR00T N1.5 foundation model for robot policy inference and control.

- **Simulation Environment**: A modified RoboCasa simulation framework that executes specified tasks from given initial world states.

## Recommended Prerequisites

- **Operating System**: Ubuntu 20.04 or 22.04
- **Python**: 3.10
- **CUDA**: 12.4 (recommended) or 11.8
- **GPU**: H100, L40, RTX 4090, A6000, or RTX 3090 (for robot controller)
- **System Dependencies**: `ffmpeg`, `libsm6`, `libxext6`
- **Conda**: For environment management

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/LIIHWF/robot-testing-sc
cd robot-testing-sc
```

### 2. Setup Abstract Layer

The abstract layer requires Python dependencies for grammar parsing, Z3 SMT solving, and ASP enumeration.

```bash
# Create conda environment
conda create -n robot-testing python=3.10
conda activate robot-testing

# Install Python dependencies
pip install z3-solver clingo ply

# Install Prolog dependencies (SWI-Prolog)
sudo apt-get update
sudo apt-get install swi-prolog
```

### 3. Setup Concrete Layer

The concrete layer requires additional dependencies for simulation execution and falsification.

```bash
# Ensure you're in the robot-testing conda environment
conda activate robot-testing

# Install falsification dependencies
pip install rtamt nevergrad numpy scipy

# Install simulation interface dependencies
pip install pyzmq requests
```

### 4. Setup Robot Controller (GR00T N1.5)

The robot controller requires the GR00T N1.5 model and its dependencies.

```bash
# Navigate to robot_controller directory
cd robot_controller

# Install GR00T dependencies
pip install --upgrade setuptools
pip install -e .[base]
pip install --no-build-isolation flash-attn==2.7.1.post4

# Download the GR00T N1.5 model using the download script
pip install huggingface_hub
python download.py
```

### 5. Setup Simulation Environment

This is a modified version of [robocasa-gr1-tabletop-tasks](https://github.com/robocasa/robocasa-gr1-tabletop-tasks).

```bash
# 1. Using created conda environment
conda activate robot-testing

# 2. Install robosuite
pip install -e robosuite_gr1

# 3.Install robocasa
pip install -e robocasa_gr1

# 4. Download assets
cd robocasa_gr1
python robocasa/scripts/download_tabletop_assets.py -y
```

## Usage

### Scripts Overview

The `scripts/` directory contains three main execution scripts:

1. **`run_abstract_gen.sh`**: Generates abstract test configurations from grammar
2. **`run_falsification.sh`**: Executes falsification tests on generated configurations
3. **`start_robot.sh`**: Starts the robot inference service

### Complete Workflow Example

```bash
# 1. Generate abstract test configurations
conda activate robot-testing
./scripts/run_abstract_gen.sh -d 4 -t 2

# 2. Start robot inference service
conda activate robot-testing
./scripts/start_robot.sh

# 3. Run falsification tests in another terminal
conda-activate robot-testing
./scripts/run_falsification.sh --all -c generated_data/abstract_layer/2-way_ct_configurations_trans.json

# Falsification results are saved to generated_data/falsify_results
```

### 1. Generate Abstract Test Configurations

The `run_abstract_gen.sh` script generates abstract test configurations through a multi-step pipeline:

```bash
./scripts/run_abstract_gen.sh [OPTIONS]
```

**Options**:
- `-o OUTPUT_DIR`: Output directory for generated data (default: `generated_data/abstract_layer`)
- `-d DEPTH`: Maximum depth for task enumeration (default: 4)
- `-t T_WAY`: T-way coverage level (default: 2)

**What it does**:
1. Enumerates tasks and builds ACTS model from grammar
2. Analyzes weakest preconditions of generated tasks
3. Generates tabletop model
4. Combines models into unified CT model
5. Simplifies the combined model
6. Enumerates candidate configurations via ASP
7. Selects T-way covering configuration set
8. Translates configurations to target format
9. Verifies resulting configurations

**Example**:
```bash
# Generate 2-way covering configurations with depth 5
./scripts/run_abstract_gen.sh -o my_output -d 5 -t 2

# Output files:
# - my_output/2-way_ct_configurations_trans.json (translated configurations)
# - my_output/combined_model_all_configs_trans.json (all configurations)
```

### 2. Run Falsification Tests

The `run_falsification.sh` script executes falsification tests on generated configurations:

**Options**:
- `-c, --config-path PATH`: Path to configuration JSON file (default: `generated_data/abstract_layer/2-way_ct_configurations_trans.json`)
- `-b, --budget N`: Budget for each falsification run (default: 5)
- `-o, --output DIR`: Output directory for results and videos (default: `falsify_results`)
- `-a, --all`: Falsify all configurations in the file
- `-n N`: Falsify the Nth configuration
- `-h, --help`: Show help message

**What it does**:
- Automatically counts the number of configurations in the JSON file
- Creates the output directory if it doesn't exist
- For each selected configuration:
  - Instantiates concrete test parameters
  - Executes simulation with the configuration
  - Performs STL-based falsification
  - Saves results and videos to the specified output directory
  - Reports violations if found

**Examples**:
```bash
# Falsify all configurations with default settings
./scripts/run_falsification.sh --all

# Falsify the 10th configuration
./scripts/run_falsification.sh -n 10

# Falsify all configs from a custom file with custom budget
./scripts/run_falsification.sh -c my_config.json --all --budget 10

# Falsify a specific configuration with custom settings and output directory
./scripts/run_falsification.sh -c my_config.json -n 5 --budget 20 -o my_results

# Falsify all configs and save to custom output directory
./scripts/run_falsification.sh --all -o my_results
```

### 3. Start Robot Inference Service

The `start_robot.sh` script starts the GR00T N1.5 inference service:

```bash
./scripts/start_robot.sh
```

**What it does**:
- Loads the GR00T N1.5-3B model
- Starts an HTTP inference server
- Accepts observation inputs and returns action predictions

**Configuration**:
- Model path: `./robot_controller/models/GR00T-N1.5-3B` (default)
- Server mode: HTTP server for inference requests

**Example**:
```bash
# Start the inference service
./scripts/start_robot.sh

# In another terminal, you can send requests to the server
python robot_controller/scripts/inference_service.py --client
```

**Note**: Make sure you have downloaded the GR00T N1.5 model checkpoint before running this script. The model will be automatically downloaded from HuggingFace if not present locally.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.

## Citation

If you use this testing framework in your research, please cite:
```
@misc{li2026validatinggeneralistrobotssituation,
      title={Validating Generalist Robots with Situation Calculus and STL Falsification}, 
      author={Changwen Li and Rongjie Yan and Chih-Hong Cheng and Jian Zhang},
      year={2026},
      eprint={2601.03038},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2601.03038}, 
}
```
