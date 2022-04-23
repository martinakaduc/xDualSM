import os
import json
import argparse
import networkx as nx
import numpy as np

from tqdm import tqdm
from random import choice, seed, shuffle
from multiprocessing import Process

RAW_DATASETS_PATH = "./raw_datasets"

def parse_args():
    parser = argparse.ArgumentParser(description='Synthetic graphs')
    parser.add_argument("--cont", default=False, type=bool, help="Continue generating")
    parser.add_argument("--num_subgraphs", default=2000, type=int, help="Number of subgraphs")
    return parser.parse_args()

def read_config(config_file):
    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def read_dataset(path, ds_name):
    ds_dir = os.path.join(path, ds_name)

    node_labels_file = open(os.path.join(ds_dir, ds_name+".node_labels"), 'r')
    edges_file = open(os.path.join(ds_dir, ds_name+".edges"), 'r')
    graph_idx_file = open(os.path.join(ds_dir, ds_name+".graph_idx"), 'r')

    total_graph = nx.Graph()
    transactions_by_nid = {}

    node_labels = node_labels_file.read().strip().split('\n')
    label_set = set(node_labels)
    label_mapping = {x: i+1 for i, x in enumerate(label_set)}
    node_labels = [(i, {'label':label_mapping[x]}) for i, x in enumerate(node_labels)]
    total_graph.add_nodes_from(node_labels)

    edges = edges_file.read().strip().split('\n')
    edges = [(int(sn)-1, int(en)-1,  {'label':1}) for line in edges for sn, en in [line.split(',')]]
    total_graph.add_edges_from(edges)

    nid_to_transaction = graph_idx_file.read().strip().split('\n')
    nid_to_transaction = {i: int(x) - 1 for i, x in enumerate(nid_to_transaction)}
    
    transaction_ids = set(nid_to_transaction.values())
    print("Processing transactions...")
    for tid in tqdm(transaction_ids):
        filtered_nid_by_transaction = list(y[0] for y in filter(lambda x: x[1] == tid, nid_to_transaction.items()))
        transactions_by_nid[tid] = filtered_nid_by_transaction

    return total_graph, transactions_by_nid

def save_per_source(graph_id, H, iso_subgraphs, noniso_subgraphs, dataset_path):
    # Ensure path
    subgraph_path = os.path.join(dataset_path, str(graph_id))
    ensure_path(subgraph_path)

    # Save source graphs
    source_graph_file = os.path.join(subgraph_path, "source.lg")
    with open(source_graph_file, 'w', encoding='utf-8') as file:
        file.write('t # {0}\n'.format(graph_id))
        for node in H.nodes:
            file.write('v {} {}\n'.format(node, H.nodes[node]['label']))
        for edge in H.edges:
            file.write('e {} {} {}\n'.format(edge[0], edge[1], H.edges[(edge[0], edge[1])]['label']))

    # Save subgraphs
    iso_subgraph_file = os.path.join(subgraph_path, "iso_subgraphs.lg")
    noniso_subgraph_file = os.path.join(subgraph_path, "noniso_subgraphs.lg")
    iso_subgraph_mapping_file = os.path.join(subgraph_path, "iso_subgraphs_mapping.lg")
    noniso_subgraph_mapping_file = os.path.join(subgraph_path, "noniso_subgraphs_mapping.lg")

    isf = open(iso_subgraph_file, "w", encoding="utf-8")
    ismf = open(iso_subgraph_mapping_file, "w", encoding="utf-8")

    for subgraph_id, S in enumerate(iso_subgraphs):
        isf.write('t # {0}\n'.format(subgraph_id))
        ismf.write('t # {0}\n'.format(subgraph_id))
        node_mapping = {}
        list_nodes = list(S.nodes)
        shuffle(list_nodes)
        
        for node_idx, node_emb in enumerate(list_nodes):
            isf.write('v {} {}\n'.format(node_idx, S.nodes[node_emb]['label']))
            ismf.write('v {} {}\n'.format(node_idx, node_emb))
            node_mapping[node_emb] = node_idx

        for edge in S.edges:
            edge_0 = node_mapping[edge[0]]
            edge_1 = node_mapping[edge[1]]
            isf.write('e {} {} {}\n'.format(edge_0, edge_1, S.edges[(edge[0], edge[1])]['label']))

    isf.close()
    ismf.close()

    nisf = open(noniso_subgraph_file, "w", encoding="utf-8")
    nismf = open(noniso_subgraph_mapping_file, "w", encoding="utf-8")
    for subgraph_id, S in enumerate(noniso_subgraphs):
        nisf.write('t # {0}\n'.format(subgraph_id))
        nismf.write('t # {0}\n'.format(subgraph_id))
        node_mapping = {}
        list_nodes = list(S.nodes)
        shuffle(list_nodes)

        for node_idx, node_emb in enumerate(list_nodes):
            nisf.write('v {} {}\n'.format(node_idx, S.nodes[node_emb]['label']))
            if not S.nodes[node_emb]['modified']:
                nismf.write('v {} {}\n'.format(node_idx, node_emb))
            node_mapping[node_emb] = node_idx

        for edge in S.edges:
            edge_0 = node_mapping[edge[0]]
            edge_1 = node_mapping[edge[1]]
            nisf.write('e {} {} {}\n'.format(edge_0, edge_1, S.edges[(edge[0], edge[1])]['label']))
    
    nisf.close()
    nismf.close()

def node_match(first_node, second_node):
        return first_node["label"] == second_node["label"]

def edge_match(first_edge, second_edge):
        return first_edge["label"] == second_edge["label"]

def generate_iso_subgraph(graph, no_of_nodes,
                          avg_degree, std_degree,
                          *args, **kwargs):
    graph_nodes = graph.number_of_nodes()
    node_ratio = no_of_nodes / graph_nodes
    if node_ratio > 1:
        node_ratio = 1

    min_edges = int(no_of_nodes * (avg_degree - std_degree) / 2)
    max_edges = int(no_of_nodes * (avg_degree + std_degree) / 2)
    subgraph = None
    iteration = 0

    while subgraph is None or subgraph.number_of_nodes() < 2 or not nx.is_connected(subgraph):
        chose_nodes = np.random.choice([0, 1], size=graph_nodes, replace=True, p=[1 - node_ratio, node_ratio])
        remove_nodes = np.where(chose_nodes == 0)[0]
        subgraph = graph.copy()
        subgraph.remove_nodes_from(remove_nodes)

        iteration += 1
        if iteration > 5:
            node_ratio *= 1.05
            if node_ratio > 1:
                node_ratio = 1
            iteration = 0

    high = subgraph.number_of_edges() - subgraph.number_of_nodes() + 2
    if high > 0:
        modify_times = np.random.randint(0, high)
        for _ in range(modify_times):
            if subgraph.number_of_edges() <= min_edges:
                break
            subgraph = remove_random_edge(subgraph)

    return subgraph

def remove_random_node(graph):
    new_graph = None

    while new_graph is None or not nx.is_connected(new_graph):
        delete_node = np.random.choice(graph.nodes)
        new_graph = graph.copy()
        new_graph.remove_node(delete_node)

    return new_graph

def remove_random_nodes(graph, num_nodes):
    while graph.number_of_nodes() > num_nodes:
        graph = remove_random_node(graph)

    return graph

def remove_random_edge(graph):
    new_graph = None
    
    while new_graph is None or not nx.is_connected(new_graph):
        delete_edge = choice(list(graph.edges))
        new_graph = graph.copy()
        new_graph.remove_edge(*delete_edge)

    return new_graph

def add_random_edges(current_graph, NE, min_edges=61, max_edges=122):
    """
    randomly adds edges between nodes with no existing edges.
    based on: https://stackoverflow.com/questions/42591549/add-and-delete-a-random-edge-in-networkx
    :param probability_of_new_connection:
    :return: None
    """
    if current_graph:
        new_edges = []
        connected = []
        for i in current_graph.nodes:
            # find the other nodes this one is connected to
            connected = connected + [to for (fr, to) in current_graph.edges(i)]
            connected = list(dict.fromkeys(connected))
            # and find the remainder of nodes, which are candidates for new edges

        unconnected = [j for j in current_graph.nodes if not j in connected]
        # print('Connected:', connected)
        # print('Unconnected', unconnected)
        is_connected = False
        while not is_connected:  # randomly add edges until the graph is connected
            if len(unconnected)==0:
                break
            new = choice(unconnected)
            old = choice(connected)
            edge_label = np.random.randint(0, NE)

            # for visualise only
            current_graph.add_edges_from([(old, new, {'label': edge_label})])
            current_graph.nodes[old]["modified"] = True
            # book-keeping, in case both add and remove done in same cycle
            unconnected.remove(new)
            connected.append(new)
            is_connected = nx.is_connected(current_graph)
            # print('Connected:', connected)
            # print('Unconnected', unconnected

        num_edges = np.random.randint(min_edges, max_edges+1)

        while current_graph.number_of_edges() < num_edges:
            old_1, old_2 = np.random.choice(connected, 2, replace=False)
            if current_graph.has_edge(old_1, old_2):
                old_1, old_2 = np.random.choice(connected, 2, replace=False)
            edge_label = np.random.randint(1, NE+1)
            current_graph.add_edges_from([(old_1, old_2, {'label': edge_label})])
            current_graph.nodes[old_1]["modified"] = True
            current_graph.nodes[old_2]["modified"] = True

    return current_graph

def add_random_nodes(graph, num_nodes, id_node_start, number_label_node, number_label_edge, min_edges, max_edges):
    graph_nodes = graph.number_of_nodes()
    number_of_possible_nodes_to_add = num_nodes - graph_nodes

    node_id = id_node_start  # start node_id from the number of nodes already in the common graph (note that the node ids are numbered from 0)
    # so if there were 5 nodes in the common graph (0,1,2,3,4) start adding new nodes from node 5 on wards
    added_nodes = []
    for i in range(number_of_possible_nodes_to_add):
        node_label = np.random.randint(1, number_label_node+1)
        added_nodes.append((node_id, {'label': node_label, 'modified': True}))
        node_id += 1

    # add all nodes to current graph
    graph.add_nodes_from(added_nodes)
    add_random_edges(graph, number_label_edge, min_edges, max_edges)
    return graph

def random_modify(graph, NN, NE, graph_nodes, min_edges, max_edges):
    num_steps = np.random.randint(1, graph.number_of_nodes() + graph.number_of_edges())
    modify_type = None

    while num_steps > 0:
        modify_type = np.random.randint(0, 3)

        if modify_type == 0: # Change node label
            chose_node = np.random.choice(graph.nodes)
            origin_label = graph.nodes[chose_node]["label"]
            new_label = np.random.randint(1, NN+1)
            while new_label == origin_label:
                new_label = np.random.randint(1, NN+1)

            graph.nodes[chose_node]["label"] = new_label
            graph.nodes[chose_node]["modified"] = True

        # elif modify_type == 1:
        #     chose_edge = np.random.choice(graph.nodes, size=2, replace=False)
        #     while not graph.has_edge(*chose_edge):
        #         chose_edge = np.random.choice(graph.nodes, size=2, replace=False)

        #     origin_label = graph[chose_edge[0]][chose_edge[1]]["label"]
        #     new_label = np.random.randint(1, NE+1)
        #     while new_label == origin_label:
        #         new_label = np.random.randint(1, NE+1)

        #     graph[chose_edge[0]][chose_edge[1]]["label"] = new_label
        #     graph.nodes[chose_edge[0]]["modified"] = True
        #     graph.nodes[chose_edge[1]]["modified"] = True

        elif modify_type == 1: # Remove & add random node
            graph = remove_random_nodes(graph, graph.number_of_nodes()-1)
            graph = add_random_nodes(graph, graph.number_of_nodes() + 1, graph_nodes,
                                    NN, NE, 
                                    min_edges, max_edges)

        elif modify_type == 2: # Remove & add random edge
            graph = remove_random_edge(graph)
            graph = add_random_edges(graph, NE, min_edges, max_edges)

        num_steps -= 1

    return graph

def generate_noniso_subgraph(graph, no_of_nodes,
                              avg_degree, std_degree,
                              number_label_node, number_label_edge,
                              *args, **kwargs):
    graph_nodes = graph.number_of_nodes()
    node_ratio = no_of_nodes / graph_nodes
    if node_ratio > 1:
        node_ratio = 1

    min_edges = int(no_of_nodes * min(no_of_nodes - 1, avg_degree - std_degree) / 2)
    max_edges = int(no_of_nodes * min(no_of_nodes - 1, avg_degree + std_degree) / 2)
    subgraph = None
    iteration = 0

    while subgraph is None or subgraph.number_of_nodes() < 2 or not nx.is_connected(subgraph):
        chose_nodes = np.random.choice([0, 1], size=graph_nodes, replace=True, p=[1 - node_ratio, node_ratio])
        remove_nodes = np.where(chose_nodes == 0)[0]
        subgraph = graph.copy()
        subgraph.remove_nodes_from(remove_nodes)

        iteration += 1
        if iteration > 5:
            node_ratio *= 1.05
            if node_ratio > 1:
                node_ratio = 1
            iteration = 0

    for nid in subgraph.nodes:
        subgraph.nodes[nid]["modified"] = False

    if subgraph.number_of_nodes() > no_of_nodes:
        subgraph = remove_random_nodes(subgraph, no_of_nodes)
    elif subgraph.number_of_nodes() < no_of_nodes:
        subgraph = add_random_nodes(subgraph, no_of_nodes, graph_nodes,
                                    number_label_node, number_label_edge, 
                                    min_edges, max_edges)

    high = subgraph.number_of_edges() - subgraph.number_of_nodes() + 2
    if high > 0:
        modify_times = np.random.randint(0, high)
        for _ in range(modify_times):
            subgraph = remove_random_edge(subgraph)

    subgraph = random_modify(subgraph, number_label_node, number_label_edge, graph_nodes, min_edges, max_edges)
    graph_matcher = nx.algorithms.isomorphism.GraphMatcher(graph, subgraph, node_match=node_match, edge_match=edge_match)
    
    while graph_matcher.subgraph_is_isomorphic():
        subgraph = random_modify(subgraph, number_label_node, number_label_edge, graph_nodes, min_edges, max_edges)
        graph_matcher = nx.algorithms.isomorphism.GraphMatcher(graph, subgraph, node_match=node_match, edge_match=edge_match)

    return subgraph

def generate_subgraphs(graph, number_subgraph_per_source,
                       *args, **kwargs):
    list_iso_subgraphs = []
    list_noniso_subgraphs = []

    for _ in tqdm(range(number_subgraph_per_source)):
        no_of_nodes = np.random.randint(2, graph.number_of_nodes() + 1)
        prob = np.random.randint(0, 2)
        if prob == 1:
            list_iso_subgraphs.append(generate_iso_subgraph(graph, no_of_nodes, *args, **kwargs))
        else:
            list_noniso_subgraphs.append(generate_noniso_subgraph(graph, no_of_nodes, *args, **kwargs))

    return list_iso_subgraphs, list_noniso_subgraphs

def generate_one_sample(idx, number_subgraph_per_source,
                        source_graphs,
                        *arg, **kwarg):
    source_graph = source_graphs[idx]
    iso_subgraphs, noniso_subgraphs = generate_subgraphs(source_graph, number_subgraph_per_source, *arg, **kwarg)

    return source_graph, iso_subgraphs, noniso_subgraphs

def generate_batch(start_idx, stop_idx, number_source, dataset_path,
                   *args, **kwargs):

    for idx in range(start_idx, stop_idx):
        print("SAMPLE %d/%d" % (idx+1, number_source))
        graph, iso_subgraphs, noniso_subgraphs = generate_one_sample(idx, *args, **kwargs)
        save_per_source(idx, graph, iso_subgraphs, noniso_subgraphs, dataset_path)

def generate_dataset(dataset_path, is_continue, number_source, *args, **kwargs):
    print("Generating...")
    list_processes = []

    if is_continue != False:
        print("Continue generating...")
        generated_sample = os.listdir(dataset_path)
        generated_sample = [int(x) for x in generated_sample]
        remaining_sample = np.array(sorted(set(range(number_source)) - set(generated_sample)))
        gap_list = remaining_sample[1:] - remaining_sample[:-1]
        gap_idx = np.where(gap_list > 1)[0] + 1
        if len(gap_idx) < 1:
            list_idx = [(remaining_sample[0], remaining_sample[-1]+1)]
        else:
            list_idx = [(remaining_sample[0], remaining_sample[gap_idx[0]])] + \
                    [(remaining_sample[gap_idx[i]], remaining_sample[gap_idx[i+1]]) for i in range(gap_idx.shape[0]-1)] + \
                    [(remaining_sample[gap_idx[-1]], remaining_sample[-1]+1)]
    
        for start_idx, stop_idx in list_idx:
            list_processes.append(Process(target=generate_batch, 
                                        args=(start_idx,stop_idx,number_source,dataset_path), 
                                        kwargs=kwargs))

    else:
        # num_process = 1
        num_process = os.cpu_count()
        batch_size = int(number_source / num_process) + 1
        start_idx = 0
        stop_idx = start_idx + batch_size

        for idx in range(num_process):
            list_processes.append(Process(target=generate_batch, 
                                        args=(start_idx,stop_idx,number_source,dataset_path), 
                                        kwargs=kwargs))

            start_idx = stop_idx
            stop_idx += batch_size
            if stop_idx > number_source:
                stop_idx = number_source

    for idx in range(len(list_processes)):
        list_processes[idx].start()

    for idx in range(len(list_processes)):
        list_processes[idx].join()

def separate_graphs(total_graph, transaction_by_id):
    separeted_graphs = {}
    for gid in transaction_by_id:
        separeted_graphs[gid] = total_graph.subgraph(transaction_by_id[gid])

    return separeted_graphs

def calculate_ds_attr(graph_ds, total_graph, num_subgraphs):
    '''
        "number_source": 1000,
        "avg_source_size": 60,
        "std_source_size": 10,
        "avg_degree": 3.5,
        "std_degree": 0.5,
        "number_label_node": 20
    '''

    attr_dict = {}
    attr_dict["number_source"] = len(graph_ds)
    list_source_node = [g.number_of_nodes() for _, g in graph_ds.items()]
    list_source_edge = [g.number_of_edges() for _, g in graph_ds.items()]

    mean_size, std_size = np.mean(list_source_node, axis=0), np.std(list_source_node, axis=0)
    attr_dict["avg_source_size"] = mean_size
    attr_dict["std_source_size"] = std_size

    list_avg_degree = [e*2/n for n, e in zip(list_source_node, list_source_edge)]
    mean_degree, std_degree = np.mean(list_avg_degree, axis=0), np.std(list_avg_degree, axis=0)
    attr_dict["avg_degree"] = mean_degree
    attr_dict["std_degree"] = std_degree
  
    total_label = [total_graph.nodes[n]["label"] for n in total_graph.nodes]
    attr_dict["number_label_node"] = len(set(total_label))
    attr_dict["number_label_edge"] = 1 # TO_REMOVE

    attr_dict["number_subgraph_per_source"] = num_subgraphs # TO_REMOVE
    return attr_dict

def save_config_for_synthesis(ds_name, configs):
    configs["number_source"] *= 4
    with open("configs/%s.json" % ds_name, "w") as f:
        json.dump(configs, f, indent=4)

def process_dataset(path, ds_name, is_continue, num_subgraphs):
    total_graph, transaction_by_nid = read_dataset(path, ds_name)

    source_graphs = separate_graphs(total_graph, transaction_by_nid)
    config = calculate_ds_attr(source_graphs, total_graph, num_subgraphs)

    del total_graph
    del transaction_by_nid

    seed(42)
    np.random.seed(42)
    dataset_path = os.path.join("datasets", 
                   os.path.basename(ds_name).split(".")[0] + "_test")
    ensure_path(dataset_path)

    generate_dataset(dataset_path=dataset_path, 
                     is_continue=is_continue,
                     source_graphs=source_graphs,
                      **config)

    save_config_for_synthesis(ds_name, config)                 

if __name__ == '__main__':
    list_datasets = os.listdir(RAW_DATASETS_PATH)
    args = parse_args()

    for dataset in list_datasets:
        # if dataset != "COX2": continue # TO_TEST

        print("PROCESSING DATASET:", dataset)
        process_dataset(path=RAW_DATASETS_PATH, ds_name=dataset, is_continue=args.cont, num_subgraphs=args.num_subgraphs)