#!/bin/bash

# Default values
CONFIG_PATH=""
BUDGET=5
NUM_CONFIGS=0
USE_ALL=false
OUTPUT_DIR="generated_data/falsify_results"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config-path|-c)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --budget|-b)
            BUDGET="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --all|-a)
            USE_ALL=true
            shift
            ;;
        -n)
            NUM_CONFIGS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 -c CONFIG_PATH [OPTIONS]"
            echo ""
            echo "Required Options:"
            echo "  -c, --config-path PATH    Path to configuration JSON file (required)"
            echo ""
            echo "Options:"
            echo "  -b, --budget N            Budget for each falsification run (default: 5)"
            echo "  -o, --output DIR          Output directory for results and videos (default: falsify_results)"
            echo "  -a, --all                 Falsify all configurations in the file"
            echo "  -n N                      Falsify the Nth configuration"
            echo "  -h, --help                Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 -c my_config.json --all                # Falsify all configs from custom file"
            echo "  $0 -c my_config.json -n 5                 # Falsify the 5th config"
            echo "  $0 -c my_config.json -n 5 --budget 10     # Falsify the 5th config with budget 10"
            echo "  $0 -c my_config.json --all -o my_results # Falsify all configs, save to my_results/"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if config path is provided
if [ -z "$CONFIG_PATH" ]; then
    echo "Error: --config-path (-c) is required"
    echo "Use --help for usage information"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Configuration file not found: $CONFIG_PATH"
    exit 1
fi

# Count total number of configurations using Python
TOTAL_CONFIGS=$(python3 -c "
import json
import sys
try:
    with open('$CONFIG_PATH', 'r') as f:
        configs = json.load(f)
        if isinstance(configs, list):
            print(len(configs))
        else:
            print(0)
except Exception as e:
    print(f'Error reading config file: {e}', file=sys.stderr)
    sys.exit(1)
")

if [ $? -ne 0 ] || [ -z "$TOTAL_CONFIGS" ]; then
    echo "Error: Failed to read configuration file"
    exit 1
fi

if [ "$TOTAL_CONFIGS" -eq 0 ]; then
    echo "Error: No configurations found in $CONFIG_PATH"
    exit 1
fi

echo "Found $TOTAL_CONFIGS configurations in $CONFIG_PATH"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR (results and videos will be saved here)"

# Determine which config(s) to falsify
if [ "$USE_ALL" = true ]; then
    START_CONFIG=1
    END_CONFIG=$TOTAL_CONFIGS
    echo "Falsifying all $TOTAL_CONFIGS configurations..."
elif [ "$NUM_CONFIGS" -gt 0 ]; then
    if [ "$NUM_CONFIGS" -gt "$TOTAL_CONFIGS" ]; then
        echo "Error: Configuration $NUM_CONFIGS does not exist. Only $TOTAL_CONFIGS configurations available."
        exit 1
    fi
    START_CONFIG=$NUM_CONFIGS
    END_CONFIG=$NUM_CONFIGS
    echo "Falsifying configuration $NUM_CONFIGS..."
else
    echo "Error: Must specify either --all or -n N"
    echo "Use --help for usage information"
    exit 1
fi

# Run falsification for each configuration
for i in $(seq $START_CONFIG $END_CONFIG); do
    echo ""
    echo "========================================="
    if [ "$USE_ALL" = true ]; then
        echo "Falsifying configuration $i/$TOTAL_CONFIGS"
    else
        echo "Falsifying configuration $i"
    fi
    echo "========================================="
    python concrete_layer/falsifier/falsify.py \
        --config-number $i \
        --config-path "$CONFIG_PATH" \
        --run \
        --budget $BUDGET \
        --output-dir "$OUTPUT_DIR"
done

echo ""
echo "========================================="
echo "Falsification complete!"
if [ "$USE_ALL" = true ]; then
    echo "Processed $TOTAL_CONFIGS configurations"
else
    echo "Processed configuration $NUM_CONFIGS"
fi
echo "========================================="
