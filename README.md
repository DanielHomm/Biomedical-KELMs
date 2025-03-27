# NLP Lab
This repository contains the code for the NLP Lab Course at TUM from Summer Semester 2024. 
The project was conducted by Johannes Burmester, Mikhail Konov, and Daniel Homm

The project mostly reuses the idea from Meng, Z., Liu, F., Clark, T. H., Shareghi, E., & Collier, N. (2021). Mixture-of-Partitions: Infusing large biomedical knowledge graphs into BERT.
However, the here included code is compatible with adapterhub.

A summary of our findings can be found in the poster.pdf.

## Biomedical KELMs

This repository contains the practical course work of the summer semester 2024.

The work in this repository is structured as follows:

In the src folder, the project is divided into three sup parts.

1.  The preprocessing to partition the knowledge graph. The partitioned knowledge graph is afterwards used to pretrain the adapters.

2. The pretraining of the adapters

3. The fine-tuning of the downstream task

   For the downstream tasks, the Python script fine_tuning_3_all.py is used to load all the data, the model, and the adapters for the different fine_tuning tasks.

   The different bash scripts include different experiments.

The data folder contains the different data sets for all downstream tasks.

The project_doc folder contains the presentations and the final report.
