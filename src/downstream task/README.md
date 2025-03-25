Currently the fine-tuning.py script can handle BioASQ and PubMedQA.

The script arguments include:
-- BASE_MODEL: BioBert: dmis-lab/biobert-v1.1, SciBert: allenai/scibert_scivocab_uncased, or PubMedBert: microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
--tokenizer: Tokenizer related to the base model
--task: PubMedQA or BioASQ
--fine_tuning_strategy: model+adapters, fusion_only, or model_only
--data_dir: Data path to the correct data set. Such that the related load function can handle the path.
--adapter_dir: Path to the adatapers e.g. for me it is ./adapters/S20Rel_ 
--adapter_name: Type of adapter, currently LP, LP_NB, EP_NB (The path for the adapters is a combination of adapter_dir + adapter_name
--lr: Learning Rate
--batch_size: Batch Size
--epochs: Epochs
--runs: How many runs you want to run.
--seq_length: The maximum sequence length for the input.
--warmup_proportion: The warmup_proportion for the training.
--patient: Patience for early stopping (currently not working since evaluation strategy is set to "no", thus evaluation will only take place after the complete training.
--gradient_accumulation_steps: The gradient accumulation step size.

The script will init at wandb to the following project: "Downstream_" + args.task

wandb.init(project="Downstream_SciBert_PubMedQa", name="Summary_model+adapters_EP_5bs_5epochs_2runs"
Example PubMedQA:
python fine_tuning.py --BASE_MODEL dmis-lab/biobert-v1.1 --task PubMedQa --fine_tuning_strategy model+adapters --data_dir ./Alex/Data/pubmedqa/  --adapter_dir ./adapters/adapters/BioBert/S20Rel_ --adapter_name EP --lr 1e-5 --batch_size 6 --epochs 5 --runs
 3 --seq_length 512 --warmup_proportion 0.1 --patient 5

Example BioASQ:
python fine_tuning.py --task BioASQ --fine_tuning_strategy fusion_only --data_dir ./Alex/Data/BioASQ/  --adapter_dir ./adapters/S20Rel_ --adapter_name LP_NB --lr 1e-5 --batch_size 5 --epochs 25 --runs
 10 --seq_length 512 --warmup_proportion 0.1 --patient 5
