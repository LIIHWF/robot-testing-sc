# Default values
DATA_DIR="generated_data/abstract_layer"
DEPTH=4
T=2

# Parse options
while getopts "o:d:t:" opt; do
  case $opt in
    o) DATA_DIR="$OPTARG";;
    d) DEPTH="$OPTARG";;
    t) T="$OPTARG";;
    \?) echo "Usage: $0 [-o output_dir] [-d depth] [-t t_way]"; exit 1;;
  esac
done

[ -d ${DATA_DIR} ] || mkdir -p ${DATA_DIR}
INTERMEDIATE_DIR="${DATA_DIR}/intermediate"
[ -d ${INTERMEDIATE_DIR} ] || mkdir -p ${INTERMEDIATE_DIR}

# 1) Enumerate tasks & build ACTS model from grammar
python3 abstract_layer/py_tools/enumeration/task_enumerator.py \
    meta_model/task_grammar.txt \
    -m ${INTERMEDIATE_DIR}/grammar_acts_model.txt \
    -o ${INTERMEDIATE_DIR}/grammar_tasks.txt \
    -d ${DEPTH}

# 2) Analyze weakest preconditions of the generated tasks
python3 abstract_layer/py_tools/analysis/wp_analysis.py \
    ${INTERMEDIATE_DIR}/grammar_tasks.txt \
    --output ${INTERMEDIATE_DIR}/wp_analysis_result.txt

# 3) Generate tabletop model
python3 meta_model/ct_model.py \
    --output ${INTERMEDIATE_DIR}/tabletop_model.txt

# 4) Combine grammar ACTS model, tabletop model, and WP analysis into a unified CT model
python3 abstract_layer/py_tools/modeling/combined_model_with_z3.py \
    ${INTERMEDIATE_DIR}/grammar_acts_model.txt \
    ${INTERMEDIATE_DIR}/tabletop_model.txt \
    ${INTERMEDIATE_DIR}/wp_analysis_result.txt \
    ${INTERMEDIATE_DIR}/combined_ct_model.txt

# 5) Simplify the combined CT model
python3 abstract_layer/py_tools/modeling/model_simplifier_eda.py \
    ${INTERMEDIATE_DIR}/combined_ct_model.txt \
    ${INTERMEDIATE_DIR}/combined_model_simplified.txt

# 6) Enumerate candidate configurations (solutions) from the simplified model via ASP
python3 abstract_layer/py_tools/enumeration/acts_to_asp_enumerator.py \
    ${INTERMEDIATE_DIR}/combined_model_simplified.txt \
    --skip-simple-eq \
    -n 100000 \
    -o ${INTERMEDIATE_DIR}/combined_model_all_configs.json

# 7) Quickly select a good 2-way covering configuration set (daily use)
python3 abstract_layer/py_tools/ct_model/greedy_tway_selection.py \
    ${INTERMEDIATE_DIR}/combined_model_all_configs.json \
    -t ${T} \
    -o ${INTERMEDIATE_DIR}/${T}-way_ct_configurations.json

# 8) Translate the selected test configurations into the target format
python3 abstract_layer/py_tools/translation/translate_ct_configurations.py \
    ${INTERMEDIATE_DIR}/${T}-way_ct_configurations.json \
    -o ${DATA_DIR}/${T}-way_ct_configurations_trans.json

# 8) Translate the selected test configurations into the target format
python3 abstract_layer/py_tools/translation/translate_ct_configurations.py \
    ${INTERMEDIATE_DIR}/combined_model_all_configs.json \
    -o ${DATA_DIR}/combined_model_all_configs_trans.json

# 9) Verify the resulting test configurations
python3 abstract_layer/py_tools/ct_model/verify_ct_config.py \
    ${DATA_DIR}/${T}-way_ct_configurations_trans.json \
    --output ${INTERMEDIATE_DIR}/verify_results.json

printf "\n\033[0;32mAbstract configurations for all tasks have been generated successfully in ${DATA_DIR}/combined_model_all_configs_trans.json\033[0m\n"
printf "\033[0;32mAbstract configurations for ${T}-way coverage have been generated successfully in ${DATA_DIR}/${T}-way_ct_configurations_trans.json\033[0m\n"
