import os
import time

import pymetis
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm.auto import tqdm

def _construct_adj(edges, num_data):
    """
        Creates a symmetric adjacency matrix from edge list data.

        Args:
            edges (numpy.ndarray): Array of edges, where each row represents a pair of connected nodes.
            num_data (int): Total number of nodes in the graph.

        Returns:
            scipy.sparse.csr_matrix: Symmetric adjacency matrix in CSR format.

        Code copied from MoP
    """
    adj = sp.csr_matrix(
        (np.ones((edges.shape[0]), dtype=np.float32), (edges[:, 0], edges[:, 1])),
        shape=(num_data, num_data),
    )
    adj += adj.transpose()
    return adj


def partition_graph(adj, idx_nodes, num_clusters):
    """
        Partitions a graph into `num_clusters` using the METIS algorithm.

        Args:
            adj (scipy.sparse.csr_matrix): Adjacency matrix of the graph.
            idx_nodes (list of int): Node indices to be partitioned.
            num_clusters (int): Number of clusters to create.

        Returns:
            part_adj (scipy.sparse.csr_matrix): Adjacency matrix of the partitioned subgraph.
            parts (list of list of int): List of node indices for each cluster.

        Code copied from MoP
    """

    start_time = time.time()
    num_nodes = len(idx_nodes)
    num_all_nodes = adj.shape[0]

    neighbor_intervals = []
    neighbors = []
    edge_cnt = 0
    neighbor_intervals.append(0)
    train_adj_lil = adj[idx_nodes, :][:, idx_nodes].tolil()
    train_ord_map = dict()
    train_adj_lists = [[] for _ in range(num_nodes)]
    for i in tqdm(range(num_nodes)):
        rows = train_adj_lil[i].rows[0]
        # self-edge needs to be removed for valid format of METIS
        if i in rows:
            rows.remove(i)
        train_adj_lists[i] = rows
        neighbors += rows
        edge_cnt += len(rows)
        neighbor_intervals.append(edge_cnt)
        train_ord_map[idx_nodes[i]] = i

    if num_clusters > 1:
        _, groups = pymetis.part_graph(nparts=num_clusters, adjacency=train_adj_lists)
    else:
        groups = [0] * num_nodes

    part_row = []
    part_col = []
    part_data = []
    parts = [[] for _ in range(num_clusters)]
    for nd_idx in tqdm(range(num_nodes)):
        gp_idx = groups[nd_idx]
        nd_orig_idx = idx_nodes[nd_idx]
        parts[gp_idx].append(nd_orig_idx)
        for nb_orig_idx in adj[nd_orig_idx].indices:
            nb_idx = train_ord_map[nb_orig_idx]
            if groups[nb_idx] == gp_idx:
                part_data.append(1)
                part_row.append(nd_orig_idx)
                part_col.append(nb_orig_idx)
    part_data.append(0)
    part_row.append(num_all_nodes - 1)
    part_col.append(num_all_nodes - 1)
    part_adj = sp.coo_matrix((part_data, (part_row, part_col))).tocsr()

    print("Partitioning done. %f seconds.", time.time() - start_time)
    return part_adj, parts


def create_partition_file(data_dir, n_partition=20, sub_set=1):
    """
    Creates a partition.txt file from the provided data files.

    Args:
        data_dir (str): The directory where the data files are located.
        n_partition (int): The number of partitions to create.
        sub_set (float): The subset of data to use (1 means full data).

    Code extracted from MoP
    """

    tri_file = os.path.join(data_dir, "train2id.txt")
    ent_file = os.path.join(data_dir, "entity2id.txt")
    # rel_file = os.path.join(data_dir, "relation2id.txt")
    partition_file = os.path.join(data_dir, f"partition_{n_partition}.txt")

    # Load entity data
    with open(ent_file, "r") as f:
        ent_total = int(f.readline())
        if sub_set != 1:
            ent_total = int(ent_total * sub_set)

    # Load triple data
    h_list = []
    t_list = []
    r_list = []
    with open(tri_file, "r") as f:
        print(f"loading triples {f.readline()}")
        for line in f.readlines():
            h, t, r = line.split("\t")
            h_list.append(int(h.strip()))
            t_list.append(int(t.strip()))
            r_list.append(int(r.strip()))

    triple_df = pd.DataFrame(
        {
            "head_id": h_list,
            "relation_id": r_list,
            "tail_id": t_list,
        }
    )

    edge_list = []
    for _, row in triple_df.iterrows():
        edge_list.append([row.head_id, row.tail_id])

    edge_list_ar = np.array(edge_list)
    num_nodes = ent_total
    adj = _construct_adj(edge_list_ar, num_nodes)
    idx_nodes = list(range(ent_total))

    # Partition the graph
    part_adj, parts = partition_graph(adj, idx_nodes, n_partition)

    # Save partitions to the partition file
    with open(partition_file, "w") as f:
        for node_list in parts:
            f.write("\t".join([str(i) for i in node_list]) + "\n")

    print(f"Partition file created at {partition_file}")