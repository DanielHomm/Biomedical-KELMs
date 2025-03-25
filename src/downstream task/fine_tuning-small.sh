#!/bin/bash

# Global variables
base_data_dir="../../../../Code/Alex/Data/"
base_adapter_dir="../../../../adapters/adapters/"

epochs=25
batch_size_pubmedqa = 12
batch_size_bioasq = 8
batch_size_hoc = 16
batch_size_mednli = 16
batch_size_medqa = 12
learning_rate=1e-5
run_pubmedqa = 10
run_bioasq = 10
run_hoc = 5
run_mednli = 3
run_medqa = 3
seq_length=512
warmup_proportion=0.1
patient=5
gradient_accumulation_steps=1



# Array of argument sets
# TASKS: BioASQ, PubMedQa, HoC, MedNLI, MedQA
# BASe MODELS: dmis-lab/biobert-v1.1, allenai/scibert_scivocab_uncased, microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
# FINE TUNING STRATEGIES: model+adapters, only_model, fusion_only
# Adapter Type: EP, EP_NB, LP, LP_NB, LP_FULL

declare -a argument_sets=(
    "allenai/scibert_scivocab_uncased allenai/scibert_scivocab_uncased PubMedQa model+adapter lp_full"
)

# Loop over the argument sets and execute the Python script with each set of arguments
for args in "${argument_sets[@]}"; do
    # Convert the string of arguments into an array
    IFS=' ' read -r -a arg_array <<< "$args"

    # Set the adapter directory based on the base model
    if [[ "${arg_array[0]}" == *"biobert"* ]]; then
        adapter_dir="${base_adapter_dir}BioBERT/S20Rel_"
    elif [[ "${arg_array[0]}" == *"scibert"* ]]; then
        adapter_dir="${base_adapter_dir}SciBERT/S20Rel_"
    # Add more base models here if needed
    elif [[ "${arg_array[0]}" == *"BiomedBERT"* ]]; then
        adapter_dir="${base_adapter_dir}PubMedBERT/S20Rel_"
    else
        echo "Unknown base model: ${arg_array[0]}"
        exit 0
    fi

    # Set the data directory based on the task
    case_arg=$(echo "${arg_array[2]}" | tr '[:upper:]' '[:lower:]')

    case "${case_arg}" in
        "bioasq")
            data_dir="${base_data_dir}BioASQ/"
            batch_size = $batch_size_bioasq
            runs = $run_bioasq
            ;;
        "pubmedqa")
            data_dir="${base_data_dir}PubMedQa/"
            batch_size = $batch_size_pubmedqa
            runs = $run_pubmedqa
            ;;
        "hoc")
            data_dir="${base_data_dir}HoC/"
            batch_size = $batch_size_hoc
            runs = $run_hoc
            ;;
        "mednli")
            data_dir="${base_data_dir}mednli/"
            batch_size = $batch_size_mednli
            runs = $run_mednli
            ;;
        "medqa")
            data_dir="${base_data_dir}medqa/"
            batch_size = $batch_size_medqa
            runs = $run_medqa
            ;;
        *)
            echo "Unknown: ${arg_array[2]}"
            exit 1
            ;;
    esac

    # Construct the command with the global variables and the current set of arguments
    command=("python" "fine_tuning_3_all.py"
        "--BASE_MODEL" "${arg_array[0]}"
        "--TOKENIZER" "${arg_array[1]}"
        "--task" "${arg_array[2]}"
        "--fine_tuning_strategy" "${arg_array[3]}"
        "--data_dir" "$data_dir"
        "--adapter_dir" "$adapter_dir"
        "--lr" "$learning_rate"
        "--batch_size" "$batch_size"
        "--epochs" "$epochs"
        "--runs" "$runs"
        "--seq_length" "$seq_length"
        "--warmup_proportion" "$warmup_proportion"
        "--patient" "$patient"
        "--gradient_accumulation_steps" "$gradient_accumulation_steps"
    )
    
    # If adapter_name is empty, remove the argument
    if [ -n "${arg_array[4]}" ]; then
        command+=("--adapter_name" "${arg_array[4]}")
    fi

    # Print the command for debugging purposes
    echo "Running command: ${command[@]}"

    # Execute the command
    "${command[@]}"

    # Check the exit status of the command
    if [ $? -ne 0 ]; then
        echo "Error executing command: ${command[@]}!!!"
    fi
done
