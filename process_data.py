# %%
# Create sample train and test keys
import sys
data_name = sys.argv[1]
data_dir = "data_%s/datasets/%s" % (data_name, sys.argv[2])
data_proccessed_dir = "data_processed/%s" % data_name

import os

# print(list_source[:5])

# %%
import networkx as nx

def read_graphs(database_file_name):
    graphs = dict()
    max_size = 0
    with open(database_file_name, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
        tgraph, graph_cnt = None, 0
        graph_size = 0
        for i, line in enumerate(lines):
            cols = line.split(' ')
            if cols[0] == 't':
                if tgraph is not None:
                    graphs[graph_cnt] = tgraph
                    if max_size < graph_size:
                        max_size = graph_size
                    graph_size = 0
                    tgraph = None
                if cols[-1] == '-1':
                    break

                tgraph = nx.Graph()
                graph_cnt = int(cols[2])

            elif cols[0] == 'v':
                tgraph.add_node(int(cols[1]), label=int(cols[2]))
                graph_size += 1

            elif cols[0] == 'e':
                tgraph.add_edge(int(cols[1]), int(cols[2]), label=int(cols[3]))

        # adapt to input files that do not end with 't # -1'
        if tgraph is not None:
            graphs[graph_cnt] = tgraph
            if max_size < graph_size:
                max_size = graph_size

    return graphs

def read_mapping(filename):
    mapping = {}
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
        tmapping, graph_cnt = None, 0
        for i, line in enumerate(lines):
            cols = line.split(' ')
            if cols[0] == 't':
                if tmapping is not None:
                    mapping[graph_cnt] = tmapping
                    
                if cols[-1] == '-1':
                    break

                tmapping = []
                graph_cnt = int(cols[2])

            elif cols[0] == 'v':
                tmapping.append((int(cols[1]), int(cols[2])))

        if tmapping is not None:
            mapping[graph_cnt] = tmapping

    return mapping

def load_graph_data(data_dir, source_id):
    
    source_graph = read_graphs("%s/%s/source.lg" % (data_dir, source_id))[int(source_id)]
    iso_subgraphs = read_graphs("%s/%s/iso_subgraphs.lg" % (data_dir, source_id))
    noniso_subgraphs = read_graphs("%s/%s/noniso_subgraphs.lg" % (data_dir, source_id))
    iso_subgraphs_mapping = read_mapping("%s/%s/iso_subgraphs_mapping.lg" % (data_dir, source_id))
    noniso_subgraphs_mapping = read_mapping("%s/%s/noniso_subgraphs_mapping.lg" % (data_dir, source_id))
    return source_graph, iso_subgraphs, noniso_subgraphs, iso_subgraphs_mapping, noniso_subgraphs_mapping

# %%
import pickle
from tqdm import tqdm

# Load and save
def load_dataset(data_dir, list_source, save_dir, additional_tag=""):
    for source_id in tqdm(list_source):
        graph, iso_subgraphs, noniso_subgraphs, \
            iso_subgraphs_mapping, noniso_subgraphs_mapping = load_graph_data(data_dir, source_id)
        
        for key, data in iso_subgraphs.items():
            with open("%s/%s_%d_iso_%s" % (save_dir, source_id, key, additional_tag), 'wb') as f:
                pickle.dump([data, graph, iso_subgraphs_mapping[key]], f)
        
        for key, data in noniso_subgraphs.items():
            with open("%s/%s_%d_non_%s" % (save_dir, source_id, key, additional_tag), 'wb') as f:
                pickle.dump([data, graph, noniso_subgraphs_mapping[key]], f)

# Load data
if not os.path.exists(data_proccessed_dir):
        os.mkdir(data_proccessed_dir)

list_source = os.listdir(data_dir)
list_source = list(filter(lambda x: os.path.isdir(os.path.join(data_dir, x)), list_source))

load_dataset(data_dir, list_source, data_proccessed_dir)

# %%
if "synthesis" in data_dir:
    # Split train test
    from sklearn.model_selection import train_test_split
    train_source, test_source = train_test_split(list_source, test_size=0.2, random_state=42)
    valid_keys = os.listdir(data_proccessed_dir)

    train_keys = [k for k in valid_keys if k.split('_')[0] in train_source]    
    test_keys = [k for k in valid_keys if k.split('_')[0] in test_source]  

    # print(train_keys[:5])
    # print(test_keys[:5])
elif "real" in data_dir:
    test_keys = os.listdir(data_proccessed_dir)
    
    data_dir_train = data_dir + '/train '
    list_source_train = os.listdir(data_dir_train)
    list_source_train = list(filter(lambda x: os.path.isdir(os.path.join(data_dir_train, x)), list_source_train))

    load_dataset(data_dir_train, list_source_train, data_proccessed_dir, additional_tag="train")

    train_keys = list(set(os.listdir(data_proccessed_dir)) - set(test_keys))

# Notice that key which has "iso" is isomorphism, otherwise non-isomorphism

# %%
# Save train and test keys
import pickle

with open("%s/train_keys.pkl"%data_proccessed_dir, 'wb') as f:
    pickle.dump(train_keys, f)
    
with open("%s/test_keys.pkl"%data_proccessed_dir, 'wb') as f:
    pickle.dump(test_keys, f)


