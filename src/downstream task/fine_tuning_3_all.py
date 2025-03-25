import numpy as np
import wandb
import json
import time
import torch
import pandas as pd
import os
import argparse
import ast
from typing import Optional, Union

import adapters
from adapters.composition import Fuse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMultipleChoice
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import get_linear_schedule_with_warmup
from transformers import TrainingArguments, Trainer, BertConfig, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import datasets
import evaluate
import gc


def load_pubmedqa(data_dir="../../data/pubmedqa/", fold_num=0):
    """
    This function loads the PubMedQA datasets from the specified directory.
    args:
        data_dir: str: the path to the directory where the PubMedQA datasets are stored
        fold_num: int: the fold number to be loaded
    returns:
        train_df: pd.DataFrame: the training dataset
        dev_df: pd.DataFrame: the development dataset
        test_df: pd.DataFrame: the test dataset
    """
    train_json = json.load(open(f"{data_dir}/pqal_fold{fold_num}/train_set.json"))
    dev_json = json.load(open(f"{data_dir}/pqal_fold{fold_num}/dev_set.json"))
    test_json = json.load(open(f"{data_dir}/test_set.json"))

    id_li, question_li, context_li, label_li = [], [], [], []
    for k, v in train_json.items():
        id_li.append(k)
        question_li.append(v["QUESTION"])
        context_li.append(v["CONTEXTS"])
        label_li.append(v["final_decision"])
    train_df = pd.DataFrame({"id": id_li, "question": question_li, "context": context_li, "label": label_li})

    dev_id_li, dev_question_li, dev_context_li, dev_label_li = [], [], [], []
    for k, v in dev_json.items():
        dev_id_li.append(k)
        dev_question_li.append(v["QUESTION"])
        dev_context_li.append(v["CONTEXTS"])
        dev_label_li.append(v["final_decision"])
    dev_df = pd.DataFrame({"id": dev_id_li, "question": dev_question_li, "context": dev_context_li, "label": dev_label_li})

    test_id_li, test_question_li, test_context_li, test_label_li = [], [], [], []
    for k, v in test_json.items():
        test_id_li.append(k)
        test_question_li.append(v["QUESTION"])
        test_context_li.append(v["CONTEXTS"])
        test_label_li.append(v["final_decision"])
    test_df = pd.DataFrame({"id": test_id_li, "question": test_question_li, "context": test_context_li, "label": test_label_li})

    print(f"Load pubmed_qa_l datasets train_df({len(train_df.index)}),dev_df({len(dev_df.index)}),test_df({len(test_df.index)})")

    for df in [train_df, dev_df, test_df]:
        df['question'] = df['question'].astype(str)
        df['context'] = df['context'].astype(str)

    train = datasets.Dataset.from_pandas(train_df).class_encode_column("label")
    eval = datasets.Dataset.from_pandas(dev_df).class_encode_column("label")
    test = datasets.Dataset.from_pandas(test_df).class_encode_column("label")

    return train, eval, test


def load_bioasq(data_dir="../../data/BioASQ/", fold_num=0):
    """
    This function loads the BioASQ datasets from the specified directory.
    args:
        data_dir: str: the path to the directory where the BioASQ datasets are stored
        fold_num: int: the fold number to be loaded
    returns:
        train_df: pd.DataFrame: the training dataset
        dev_df: pd.DataFrame: the development dataset
        test_df: pd.DataFrame: the test dataset
    """
    train_df = pd.read_csv(f"{data_dir}fold_{fold_num}/train.tsv", sep="\t")
    val_df = pd.read_csv(f"{data_dir}fold_{fold_num}/val.tsv", sep="\t")
    test_df = pd.read_csv(f"{data_dir}test.tsv", sep="\t")

    train_df['text_a'] = train_df['text_a'].astype(str)
    train_df['text_b'] = train_df['text_b'].astype(str)
    val_df['text_a'] = val_df['text_a'].astype(str)
    val_df['text_b'] = val_df['text_b'].astype(str)
    test_df['text_a'] = test_df['text_a'].astype(str)
    test_df['text_b'] = test_df['text_b'].astype(str)

    train = datasets.Dataset.from_pandas(train_df)
    train = train.class_encode_column("label")
    eval = datasets.Dataset.from_pandas(val_df)
    eval = eval.class_encode_column("label")
    test = datasets.Dataset.from_pandas(test_df)
    test = test.class_encode_column("label")

    return train, eval, test


def load_hoc(data_dir="../../data/HoC/", fold_num=0):
    """
    This function loads the HOC datasets from the specified directory.

    args:
        data_dir: str: the path to the directory where the HOC datasets are stored
        fold_num: int: the fold number to be loaded

    returns:
        train_df: pd.DataFrame: the training dataset
        dev_df: pd.DataFrame: the development dataset
        test_df: pd.DataFrame: the test dataset
    """
    train_df = pd.read_csv(f"{data_dir}fold_{fold_num}/train.tsv", sep="\t")
    val_df = pd.read_csv(f"{data_dir}fold_{fold_num}/val.tsv", sep="\t")
    test_df = pd.read_csv(f"{data_dir}test.tsv", sep="\t")

    train_df['labels'] = train_df['labels'].apply(lambda x: ast.literal_eval(x))
    val_df['labels'] = val_df['labels'].apply(lambda x: ast.literal_eval(x))
    test_df['labels'] = test_df['labels'].apply(lambda x: ast.literal_eval(x))

    train = datasets.Dataset.from_pandas(train_df)
    dev = datasets.Dataset.from_pandas(val_df)
    test = datasets.Dataset.from_pandas(test_df)

    return train, dev, test


def load_mednli(data_dir="../../data/mednli/", fold_num=0):
    """
    This function loads the MedNLI datasets from the specified directory.
    args:
        data_dir: str: the path to the directory where the MedNLI datasets are stored
        fold_num: int: the fold number to be loaded 
    
    returns:
        train_df: pd.DataFrame: the training dataset
        dev_df: pd.DataFrame: the development dataset
        test_df: pd.DataFrame: the test dataset
    """
    train_df = pd.read_csv(f"{data_dir}fold_{fold_num}/train.tsv", sep="\t")
    val_df = pd.read_csv(f"{data_dir}fold_{fold_num}/val.tsv", sep="\t")
    test_df = pd.read_csv(f"{data_dir}test.tsv", sep="\t")

    train_df['text_a'] = train_df['text_a'].astype(str)
    train_df['text_b'] = train_df['text_b'].astype(str)
    val_df['text_a'] = val_df['text_a'].astype(str)
    val_df['text_b'] = val_df['text_b'].astype(str)
    test_df['text_a'] = test_df['text_a'].astype(str)
    test_df['text_b'] = test_df['text_b'].astype(str)

    train = datasets.Dataset.from_pandas(train_df)
    train = train.class_encode_column("label")
    eval = datasets.Dataset.from_pandas(val_df)
    eval = eval.class_encode_column("label")
    test = datasets.Dataset.from_pandas(test_df)
    test = test.class_encode_column("label")

    return train, eval, test


def load_medqa(data_dir="../../data/medqa/", fold_num=0):
    train_df = pd.read_csv(f"{data_dir}fold_{fold_num}/train.tsv", sep="\t")
    val_df = pd.read_csv(f"{data_dir}fold_{fold_num}/val.tsv", sep="\t")
    test_df = pd.read_csv(f"{data_dir}test.tsv", sep="\t")

    train_df['question'] = train_df['question'].astype(str)
    train_df['answer'] = train_df['answer'].astype(str)
    train_df['options'] = train_df['options'].apply(lambda x: ast.literal_eval(x))
    val_df['question'] = val_df['question'].astype(str)
    val_df['answer'] = val_df['answer'].astype(str)
    val_df['options'] = val_df['options'].apply(lambda x: ast.literal_eval(x))
    test_df['question'] = test_df['question'].astype(str)
    test_df['answer'] = test_df['answer'].astype(str)
    test_df['options'] = test_df['options'].apply(lambda x: ast.literal_eval(x))

    train = datasets.Dataset.from_pandas(train_df)
    eval = datasets.Dataset.from_pandas(val_df)
    test = datasets.Dataset.from_pandas(test_df)

    return train, eval, test


def compute_metrics(eval_pred):
    """
    This function computes the metrics for the evaluation of the model.
    args:
        eval_pred: Tuple: the tuple containing the logits and the labels
    returns:
        results: Dict: the dictionary containing the computed metrics (currently accuracy, f1, precision, and recall)
    """
    logits, labels = eval_pred
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")

    predictions = np.argmax(logits, axis=-1)
    # Compute each metric separately with the appropriate average parameter
    accuracy_result = accuracy.compute(predictions=predictions, references=labels)
    f1_result = f1.compute(predictions=predictions, references=labels, average="macro")
    precision_result = precision.compute(predictions=predictions, references=labels, average="macro")
    recall_result = recall.compute(predictions=predictions, references=labels, average="macro")
    
    # Combine results into a single dictionary
    results = {**accuracy_result, **f1_result, **precision_result, **recall_result}
    return results


def compute_metrics_ml(eval_pred, threshold=0.5):
    """
    This function computes the metrics for the evaluation of the model.
    """
    logits, labels = eval_pred

    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(logits))

    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= threshold)] = 1
    print("predictions", predictions)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(y_true=labels, y_pred=predictions, average="macro")
    precision = precision_score(y_true=labels, y_pred=predictions, average="macro")
    recall = recall_score(y_true=labels, y_pred=predictions, average="macro")

    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}


def fine_tuning(model, project_name, args, run):
    """
    This is the main function for the fine-tuning of the model.
    args:
        BASE_MODEL: str: the name of the base model to be used
        config: BertConfig: the configuration of the model
        args: argparse.Namespace: the arguments for the fine-tuning
        run: int: the run number
    returns:
        eval_results: float: the evaluation accrucay result of the last evaluation
    """
    run_name = args.fine_tuning_strategy + "_" + str(args.epochs) + "epochs_" + str(args.batch_size) + "batch size_" + "run" + str(run) + "_" + args.adapter_name
    wandb.init(project=project_name, name=run_name, reinit=True)

    if args.fine_tuning_strategy == "model+adapters" or "fusion_only":
        adapters.init(model)

    if args.fine_tuning_strategy == "model+adapters" or args.fine_tuning_strategy == "fusion_only":
        if args.adapter_name == "":
            raise KeyError("adapter_name must be specified for model+adapters or fusion_only")

        adapters_dir = args.adapter_dir + args.adapter_name
        final_adapters = []
        for adapter_dir in os.listdir(adapters_dir):
            adapter_path = os.path.join(adapters_dir, adapter_dir)
            config_path = os.path.join(adapter_path, "adapter_config.json")
            if os.path.isdir(adapter_path):
                adapter_name = adapter_path.split("partition_")[1].split("_")[0]
                final_adapters.append(adapter_name)
                model.load_adapter(adapter_path, config=config_path, load_as=adapter_name, with_head=False)
        if adapter_name != "LP_FULL":
            fusion = Fuse(*final_adapters)
            model.add_adapter_fusion(fusion)
            model.set_active_adapters(fusion)

            if args.fine_tuning_strategy == "fusion_only":
                model.train_adapter_fusion(fusion)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"device:{device}")

    param_optimizer = list(model.named_parameters())

    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_training_steps=num_train_optimization_steps,
        num_warmup_steps=args.warmup_proportion * num_train_optimization_steps,
    )

    training_args = TrainingArguments(
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        report_to="wandb",  # enable logging to W&B
        run_name=run_name,
        logging_steps=30,
        output_dir="./training_output/" + run_name,
        overwrite_output_dir=True,
        save_strategy="no",
        remove_unused_columns=False,
        evaluation_strategy="no",  # Evaluate at the end of each epoch: epoch
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=eval,
        compute_metrics=compute_metrics_ml if args.task.lower() == "hoc" else compute_metrics,
        data_collator=DataCollatorForMultipleChoice(tokenizer=TOKENIZER, max_length=512) if args.task.lower() == "medqa" else None, # medqa needs custom data_collator
        optimizers=(optimizer, scheduler),
    )

    start_time = time.time()
    trainer.train()
    elapsed_time = time.time() - start_time

    eval_results = trainer.evaluate()
    gc.collect()
    torch.cuda.empty_cache()
    wandb.finish()

    return eval_results.get("eval_accuracy"), elapsed_time


def encode_batch_bioasq(batch):
    """Encodes a batch of input data using the model tokenizer."""
    return TOKENIZER(
        batch["text_a"],
        batch["text_b"],
        max_length=180,
        truncation=True,
        padding="max_length"
    )


def encode_batch_pubmedqa(batch):
    return TOKENIZER(
        batch["question"],
        batch["context"],
        max_length=args.seq_length,
        truncation=True,
        padding="max_length"
    )


def encode_nli(batch):
  return TOKENIZER(
      batch["text_a"],
      batch["text_b"],
      max_length=180,
      truncation=True,
      padding="max_length"
  )


def encode_ml(batch):
  return TOKENIZER(
      batch["sentence"],
      max_length=180,
      truncation=True,
      padding="max_length"
  )


def preprocess_medqa(examples):
    """
    This function preprocesses the MedQA dataset.
    """
    option_letters = ["A", "B", "C", "D"]
    option2label = {o: i for i, o in enumerate(option_letters)}
    num_answers = len(option_letters)

    question = [[ques] * num_answers for ques in examples["question"]]
    options = [[ex[opt] for opt in option_letters] for ex in examples["options"]]
    label = np.vectorize(lambda x: option2label[x])(examples["answer_idx"])

    q = sum(question, [])
    o = sum(options, [])

    tokenized_examples = TOKENIZER(q, o, truncation=True, max_length=512)
    result = {k: [v[i : i + num_answers] for i in range(0, len(v), num_answers)] for k, v in tokenized_examples.items()}
    result["label"] = label
    return result


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        rel_features = [
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "label",
        ]
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])

        flattened_features = [
            [{k: v[i] for k, v in feature.items() if k in rel_features}
             for i in range(num_choices)]
            for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


if __name__ == "__main__":
    # File Arguments
    parser = argparse.ArgumentParser(description="Fine-tuning script Practicle Course KELMs in the Medical domain")

    # Hyperparameters
    parser.add_argument("--BASE_MODEL", type=str, required=True, help="The name of the pretrained base model to be used")
    parser.add_argument("--TOKENIZER", type=str, default="", help="The name of the tokenizer to be used for the pretrained base model")
    parser.add_argument("--task", type=str, required=True, help="Current supported tasks are PubMedQA and BioASQ")
    parser.add_argument("--fine_tuning_strategy", type=str, default="model+adapters", help="Sets the fine-tuning strategy. Can be either fusion_only, model+adapters, or model_only")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--adapter_dir", type=str, required=False, help="Path to the adapters directory")
    parser.add_argument("--adapter_name", type=str, default="", help="Name of the adapter to be used")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs")
    parser.add_argument("--seq_length", type=int, default=512, help="Maximal sequence length")
    parser.add_argument("--warmup_proportion", type=float, default=0.1, help="Warmup proportion")
    parser.add_argument("--patient", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    
    args = parser.parse_args()
    if args.TOKENIZER == "":
        args.TOKENIZER = args.BASE_MODEL
    
    args.adapter_name = args.adapter_name.upper()

    print(f"Arguments: {args}")

    if args.BASE_MODEL == "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext":
        base_model_name = "PubMedBert"
    elif args.BASE_MODEL == "allenai/scibert_scivocab_uncased":
        base_model_name = "SciBert"
    elif args.BASE_MODEL == "dmis-lab/biobert-v1.1":
        base_model_name = "BioBert"
    else:
        base_model_name = "Unknown"
    
    # Define hyperparameters
    BASE_MODEL = args.BASE_MODEL #"microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    TOKENIZER = AutoTokenizer.from_pretrained(args.TOKENIZER) #"emilyalsentzer/Bio_ClinicalBERT")

    project_name = "Downstream_" + base_model_name + "_" + args.task
    
    eval_accuracies = []
    elapsed_times = []
    
    # runs are used for k-cross validation 
    for run in range(args.runs):
        if args.task.lower() == "bioasq":
            train, eval, test = load_bioasq(args.data_dir, run)

            train = train.map(encode_batch_bioasq, batched=True)
            train = train.rename_column("label", "labels")
            eval = eval.map(encode_batch_bioasq, batched=True)
            eval = eval.rename_column("label", "labels")

            train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
            eval.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

            id2label = {id: label for (id, label) in enumerate(train.features["labels"].names)}
            config = BertConfig.from_pretrained(
                BASE_MODEL,
                id2label=id2label,
            )
            model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, config=config)

        elif args.task.lower() == "pubmedqa":
            train, eval, test = load_pubmedqa(args.data_dir, run)

            train = train.map(encode_batch_pubmedqa, batched=True)
            train = train.rename_column("label", "labels")

            eval = eval.map(encode_batch_pubmedqa, batched=True)
            eval = eval.rename_column("label", "labels")

            train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
            eval.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

            id2label = {id: label for (id, label) in enumerate(train.features["labels"].names)}
            config = BertConfig.from_pretrained(
                BASE_MODEL,
                id2label=id2label,
            )
            model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, config=config)

        elif args.task.lower() == "mednli":
            train, eval, test = load_mednli(args.data_dir, run)

            train = train.map(encode_nli, batched=True)
            train = train.rename_column("label", "labels")

            eval = eval.map(encode_nli, batched=True)
            eval = eval.rename_column("label", "labels")

            train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
            eval.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

            id2label = {id: label for (id, label) in enumerate(train.features["labels"].names)}
            config = BertConfig.from_pretrained(
                BASE_MODEL,
                id2label=id2label,
            )
            model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, config=config)

        elif args.task.lower() == "medqa":
            train, eval, test = load_medqa(args.data_dir, run)
            
            train = train.map(preprocess_medqa, batched=True)
            eval = eval.map(preprocess_medqa, batched=True)

            train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
            eval.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

            model = AutoModelForMultipleChoice.from_pretrained(BASE_MODEL)

        elif args.task.lower() == "hoc":
            train, eval, test = load_hoc(args.data_dir, run)

            train = train.map(encode_ml, batched=True)
            eval = eval.map(encode_ml, batched=True)

            train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
            eval.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

            config = BertConfig.from_pretrained(
                BASE_MODEL,
                problem_type="multi_label_classification",
                num_labels=10
            )
            model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, config=config)

        else:
            raise KeyError("Please provide one of the possible tasks (BioASQ, PubMedQA, HOC, MedNLI, MedQA)")

        early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=args.patient)

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

        num_train_optimization_steps = (
            int(len(train) / args.batch_size / args.gradient_accumulation_steps) * args.epochs
        )
        
        acc, elapsed_time = fine_tuning(model, project_name, args, run)
        eval_accuracies.append(acc)
        elapsed_times.append(elapsed_time)

    
    print(f"Mean accuracy: {np.mean(eval_accuracies)} , Variance in accuracy: {np.var(eval_accuracies)}, Mean time: {np.mean(elapsed_times)}")
    wandb.init(project=project_name, name="Summary_" + args.fine_tuning_strategy + "_" + args.adapter_name + "_" + str(args.batch_size) + "bs_" + str(args.epochs) + "epochs_" + str(args.runs) +"runs", reinit=True)
    wandb.log({"Mean accuracy": np.mean(eval_accuracies), "Variance in accuracy": np.var(eval_accuracies), "Mean time": np.mean(elapsed_times)})
    wandb.finish()
    