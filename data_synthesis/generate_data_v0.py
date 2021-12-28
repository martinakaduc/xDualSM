import os
import json
import argparse
import numpy as np
import networkx as nx

from tqdm import tqdm
from random import choice, seed
from multiprocessing import Process, Manager

def parse_args():
    parser = argparse.ArgumentParser(description='Synthetic graphs')
    parser.add_argument("--config", "-c", default="configs/base.json", type=str, help="Config file")
    return parser.parse_args()

def read_config(config_file):
    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def add_labels(graph, NN, NE):
    nodes = np.array(list(graph.nodes))
    edges = np.array(list(graph.edges))

    node_labels = np.random.randint(0, NN, len(nodes)).tolist()
    edge_labels = np.random.randint(0, NE, len(edges)).tolist()

    labelled_nodes = [(nodes[k], {'label': node_labels[k], 'color': 'green'}) for k in range(len(nodes))]
    labelled_edges = [(edges[k][0], edges[k][1], {'label': edge_labels[k], 'color': 'green'}) for k in range(len(edges))]

    G = nx.Graph()
    G.add_nodes_from(labelled_nodes)
    G.add_edges_from(labelled_edges)

    return G

def generate_iso_subgraph(graph, no_of_nodes,
                           avg_degree, std_degree,
                           number_label_node, number_label_edge):
    graph_nodes = graph.number_of_nodes()
    node_ratio = no_of_nodes / graph_nodes
    if node_ratio > 1:
        node_ratio = 1

    subgraph = None
    iteration = 0

    while subgraph is None or not nx.is_connected(subgraph):
        chose_nodes = np.random.choice([0, 1], size=graph_nodes, replace=True, p=[1 - node_ratio, node_ratio])
        remove_nodes = np.where(chose_nodes == 0)[0]
        subgraph = graph.copy()
        subgraph.remove_nodes_from(remove_nodes)

        iteration += 1
        if iteration > 5:
            node_ratio *= 1.05
            iteration = 0

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
        delete_edge = np.random.choice(graph.nodes, size=2, replace=False)
        while not graph.has_edge(*delete_edge):
            delete_edge = np.random.choice(graph.nodes, size=2, replace=False)

        new_graph = graph.copy()
        new_graph.remove_edge(*delete_edge)

    return new_graph

def add_random_edges(current_graph, NE, min_edges=61, max_edges=122, nuprobability_of_new_connection=1):
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
            edge_label = np.random.randint(0, NE)

            # for visualise only
            current_graph.add_edges_from([(choice(connected), new, {'label': edge_label, 'color': 'blue'})])
            # book-keeping, in case both add and remove done in same cycle
            unconnected.remove(new)
            connected.append(new)
            is_connected = nx.is_connected(current_graph)
            # print('Connected:', connected)
            # print('Unconnected', unconnected

        num_edges = np.random.randint(min_edges, max_edges)

        while current_graph.number_of_edges() < num_edges:
            edge_label = np.random.randint(0, NE)
            current_graph.add_edges_from([(choice(connected), choice(connected), {'label': edge_label, 'color': 'blue'})])

    return current_graph

def add_random_nodes(graph, num_nodes, number_label_node, number_label_edge, min_edges, max_edges):
    graph_nodes = graph.number_of_nodes()
    number_of_possible_nodes_to_add = num_nodes - graph_nodes

    node_id = graph_nodes  # start node_id from the number of nodes already in the common graph (note that the node ids are numbered from 0)
    # so if there were 5 nodes in the common graph (0,1,2,3,4) start adding new nodes from node 5 on wards
    added_nodes = []
    for i in range(number_of_possible_nodes_to_add):
        node_label = np.random.randint(0, number_label_node)
        added_nodes.append((node_id, {'label': node_label, 'color': 'blue'}))
        node_id += 1

    # add all nodes to current graph
    graph.add_nodes_from(added_nodes)
    add_random_edges(graph, number_label_edge, min_edges, max_edges)
    return graph

def random_modify(graph, NN, NE):
    num_steps = np.random.randint(1, 10)
    while num_steps > 0:
        modify_type = np.random.randint(0, 2)
        if modify_type == 0:
            chose_node = np.random.choice(graph.nodes)
            origin_label = graph.nodes[chose_node]["label"]
            new_label = np.random.randint(0, NN)
            while new_label == origin_label:
                new_label = np.random.randint(0, NN)

            graph.nodes[chose_node]["label"] = new_label

        elif modify_type == 1:
            chose_edge = np.random.choice(graph.nodes, size=2, replace=False)
            while not graph.has_edge(*chose_edge):
                chose_edge = np.random.choice(graph.nodes, size=2, replace=False)

            origin_label = graph[chose_edge[0]][chose_edge[1]]["label"]
            new_label = np.random.randint(0, NE)
            while new_label == origin_label:
                new_label = np.random.randint(0, NE)

            graph[chose_edge[0]][chose_edge[1]]["label"] = new_label

        else:
            graph = remove_random_edge(graph)

        num_steps -= 1

    return graph

def generate_noniso_subgraph(graph, no_of_nodes, avg_subgraph_size,
                              avg_degree, std_degree,
                              number_label_node, number_label_edge):
    graph_nodes = graph.number_of_nodes()
    node_ratio = avg_subgraph_size / graph_nodes
    if node_ratio > 1:
        node_ratio = 1

    min_edges = int(no_of_nodes * (avg_degree - std_degree) / 2)
    max_edges = int(no_of_nodes * (avg_degree + std_degree) / 2)
    subgraph = None
    iteration = 0

    while subgraph is None or not nx.is_connected(subgraph):
        chose_nodes = np.random.choice([0, 1], size=graph_nodes, replace=True, p=[1 - node_ratio, node_ratio])
        remove_nodes = np.where(chose_nodes == 0)[0]
        subgraph = graph.copy()
        subgraph.remove_nodes_from(remove_nodes)

        iteration += 1
        if iteration > 5:
            node_ratio *= 1.05
            iteration = 0

    if subgraph.number_of_nodes() > no_of_nodes:
        subgraph = remove_random_nodes(subgraph, no_of_nodes)
    elif subgraph.number_of_nodes() < no_of_nodes:
        subgraph = add_random_nodes(subgraph, no_of_nodes, number_label_node, number_label_edge, min_edges, max_edges)
    else:
        subgraph = random_modify(subgraph, number_label_node, number_label_edge)

    return subgraph

def generate_subgraphs(graph, number_subgraph_per_source,
                       avg_subgraph_size, std_subgraph_size,
                       *args, **kwargs):
    list_iso_subgraphs = []
    list_noniso_subgraphs = []

    for _ in tqdm(range(number_subgraph_per_source)):
        no_of_nodes = int(np.random.normal(avg_subgraph_size, std_subgraph_size))
        if no_of_nodes < avg_subgraph_size:
            list_iso_subgraphs.append(generate_iso_subgraph(graph, no_of_nodes, *args, **kwargs))
        elif no_of_nodes > avg_subgraph_size:
            list_noniso_subgraphs.append(generate_noniso_subgraph(graph, no_of_nodes, avg_subgraph_size, *args, **kwargs))
        else:
            prob = np.random.randint(0, 1)
            if prob == 1:
                list_iso_subgraphs.append(generate_iso_subgraph(graph, no_of_nodes, *args, **kwargs))
            else:
                list_noniso_subgraphs.append(generate_noniso_subgraph(graph, no_of_nodes, avg_subgraph_size, *args, **kwargs))

    return list_iso_subgraphs, list_noniso_subgraphs

def generate_one_sample(number_subgraph_per_source,
                        avg_source_size, std_source_size,
                        avg_subgraph_size, std_subgraph_size,
                        avg_degree, std_degree,
                        number_label_node, number_label_edge):
    generated_pattern = None
    iteration = 0
    no_of_nodes = int(np.random.normal(avg_source_size, std_source_size))
    degree = np.random.normal(avg_degree, std_degree)
    probability_for_edge_creation = degree / (no_of_nodes - 1)

    while generated_pattern is None or not nx.is_connected(
                generated_pattern):  # make sure the generated graph is connected
        generated_pattern = nx.erdos_renyi_graph(no_of_nodes, probability_for_edge_creation, directed=False)
        iteration += 1
        if iteration > 5:
            probability_for_edge_creation *= 1.05
            iteration = 0

    labelled_pattern = add_labels(generated_pattern, number_label_node, number_label_edge)
    
    iso_subgraphs, noniso_subgraphs = generate_subgraphs(labelled_pattern, number_subgraph_per_source,
                                                         avg_subgraph_size, std_subgraph_size,
                                                         avg_degree, std_degree,
                                                         number_label_node, number_label_edge)
    return labelled_pattern, iso_subgraphs, noniso_subgraphs

def generate_batch(start_idx, stop_idx, number_source,
                   GRAPHS, ISO_SUBGRAPHS, NONISO_SUBGRAPHS,
                   *args, **kwargs):

    local_graphs = {}
    local_iso_subgraphs = {}
    local_noniso_subgraphs = {}

    for idx in range(start_idx, stop_idx):
        print("SAMPLE %d/%d" % (idx+1, number_source))
        graph, iso_subgraphs, noniso_subgraphs = generate_one_sample(*args, **kwargs)
        local_graphs[idx] = graph
        local_iso_subgraphs[idx] = iso_subgraphs
        local_noniso_subgraphs[idx] = noniso_subgraphs

    GRAPHS.update(local_graphs)
    ISO_SUBGRAPHS.update(local_iso_subgraphs)
    NONISO_SUBGRAPHS.update(local_noniso_subgraphs)

def generate_dataset(number_source, *args, **kwargs):
    # GRAPHS = {}
    # GRAPHS = {}
    # NONISO_SUBGRAPHS = {}

    print("Generating...")
    list_processes = []
    manager = Manager()

    GRAPHS = manager.dict()
    ISO_SUBGRAPHS = manager.dict()
    NONISO_SUBGRAPHS = manager.dict()

    batch_size = int(number_source / os.cpu_count()) + 1
    start_idx = 0
    stop_idx = start_idx + batch_size

    for idx in range(os.cpu_count()):
        list_processes.append(Process(target=generate_batch, args=(start_idx,stop_idx,number_source,
                                                             GRAPHS, ISO_SUBGRAPHS, NONISO_SUBGRAPHS), kwargs=kwargs))

        start_idx = stop_idx
        stop_idx += batch_size
        if stop_idx > number_source:
            stop_idx = number_source

    for idx in range(os.cpu_count()):
        list_processes[idx].start()

    for idx in range(os.cpu_count()):
        list_processes[idx].join()

    return GRAPHS, ISO_SUBGRAPHS, NONISO_SUBGRAPHS    

def save_dataset(graphs, iso_subgraphs, noniso_subgraphs, dataset_path):
    # Save source graphs
    source_graph_file = os.path.join(dataset_path, "source.lg")
    with open(source_graph_file, 'w', encoding='utf-8') as file:
        list_graph_id = list(sorted(graphs.keys()))
        for graph_id in list_graph_id:
            H = graphs[graph_id]
            file.write('t # {0}\n'.format(graph_id))
            for node in H.nodes:
                file.write('v {} {}\n'.format(node, H.nodes[node]['label']))
                #file.write('v {}\n'.format(node))
            for edge in H.edges:
                file.write('e {} {} {}\n'.format(edge[0], edge[1], H.edges[(edge[0], edge[1])]['label']))
                #file.write('e {} {}\n'.format(edge[0], edge[1]))

    # Save subgraphs
    for graph_id in graphs:
        subgraph_path = os.path.join(dataset_path, str(graph_id))
        ensure_path(subgraph_path)

        iso_subgraph_file = os.path.join(subgraph_path, "iso_subgraphs.lg")
        noniso_subgraph_file = os.path.join(subgraph_path, "noniso_subgraphs.lg")
        iso_subgraph_mapping_file = os.path.join(subgraph_path, "iso_subgraphs_mapping.lg")

        isf = open(iso_subgraph_file, "w", encoding="utf-8")
        ismf = open(iso_subgraph_mapping_file, "w", encoding="utf-8")

        for subgraph_id, S in enumerate(iso_subgraphs[graph_id]):
            isf.write('t # {0}\n'.format(subgraph_id))
            ismf.write('t # {0}\n'.format(subgraph_id))
            node_mapping = {}

            for node_idx, node_emb in enumerate(S.nodes):
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
        for subgraph_id, S in enumerate(noniso_subgraphs[graph_id]):
            nisf.write('t # {0}\n'.format(subgraph_id))

            for node_idx, node_emb in enumerate(S.nodes):
                nisf.write('v {} {}\n'.format(node_idx, S.nodes[node_emb]['label']))
                node_mapping[node_emb] = node_idx

            for edge in S.edges:
                edge_0 = node_mapping[edge[0]]
                edge_1 = node_mapping[edge[1]]
                nisf.write('e {} {} {}\n'.format(edge_0, edge_1, S.edges[(edge[0], edge[1])]['label']))
        
        nisf.close()

def main(config_file):
    seed(42)
    dataset_path = os.path.join("datasets", 
                   os.path.basename(config_file).split(".")[0])
    ensure_path(dataset_path)
    config = read_config(config_file)

    graphs, iso_subgraphs, noniso_subgraphs = generate_dataset(**config)
    save_dataset(graphs, iso_subgraphs, noniso_subgraphs, dataset_path)

if __name__ == '__main__':
    args = parse_args()
    main(args.config)