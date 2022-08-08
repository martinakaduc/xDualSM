import os
import utils
import torch
import time
import pickle
import argparse
import numpy as np
import networkx as nx
from gnn import gnn
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from scipy.spatial import distance_matrix
from collections import defaultdict
from dataset import BaseDataset, collate_fn, UnderSampler

def onehot_encoding_node(m, embedding_dim, is_subgraph=True):
    n = m.number_of_nodes()
    H = []
    for i in m.nodes:
        H.append(utils.node_feature(m, i, embedding_dim))
    H = np.array(H)

    # if is_subgraph:
    #     H = np.concatenate([H, np.zeros((n,embedding_dim))], 1)
    # else:
    #     H = np.concatenate([np.zeros((n,embedding_dim)), H], 1)

    return H    

class InferenceGNN():
    def __init__(self, args) -> None:
        if args.ngpu > 0:
            cmd = utils.set_cuda_visible_device(args.ngpu)
            os.environ['CUDA_VISIBLE_DEVICES'] = cmd[:-1]

        self.model = gnn(args)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = utils.initialize_model(self.model, self.device, load_save_file=args.ckpt, gpu=(args.ngpu > 0))

        self.model.eval()
        self.embedding_dim = args.embedding_dim

    def prepare_single_input(self, m1, m2):
        # Prepare subgraph
        n1 = m1.number_of_nodes()
        adj1 = nx.to_numpy_matrix(m1) + np.eye(n1)
        H1 = onehot_encoding_node(m1, self.embedding_dim, is_subgraph=True)

        # Prepare source graph
        n2 = m2.number_of_nodes()
        adj2 = nx.to_numpy_matrix(m2) + np.eye(n2)
        H2 = onehot_encoding_node(m2, self.embedding_dim, is_subgraph=False)
        
        # Aggregation node encoding
        agg_adj1 = np.zeros((n1+n2, n1+n2))
        agg_adj1[:n1, :n1] = adj1
        agg_adj1[n1:, n1:] = adj2
        agg_adj2 = np.copy(agg_adj1)
        dm = distance_matrix(H1, H2)
        dm_new = np.zeros_like(dm)
        dm_new[dm == 0.0] = 1.0
        agg_adj2[:n1,n1:] = np.copy(dm_new)
        agg_adj2[n1:,:n1] = np.copy(np.transpose(dm_new))
        
        H1 = np.concatenate([H1, np.zeros((n1,self.embedding_dim))], 1)
        H2 = np.concatenate([np.zeros((n2,self.embedding_dim)), H2], 1)
        H = np.concatenate([H1, H2], 0)

        # node indice for aggregation
        valid = np.zeros((n1+n2,))
        valid[:n1] = 1

        sample = {
                  'H':H, \
                  'A1': agg_adj1, \
                  'A2': agg_adj2, \
                  'V': valid, \
                  }

        return sample

    def input_to_tensor(self, batch_input):
        max_natoms = max([len(item['H']) for item in batch_input if item is not None])
        batch_size = len(batch_input)
    
        H = np.zeros((batch_size, max_natoms, batch_input[0]['H'].shape[-1]))
        A1 = np.zeros((batch_size, max_natoms, max_natoms))
        A2 = np.zeros((batch_size, max_natoms, max_natoms))
        V = np.zeros((batch_size, max_natoms))
        
        for i in range(batch_size):
            natom = len(batch_input[i]['H'])
            
            H[i,:natom] = batch_input[i]['H']
            A1[i,:natom,:natom] = batch_input[i]['A1']
            A2[i,:natom,:natom] = batch_input[i]['A2']
            V[i,:natom] = batch_input[i]['V']

        H = torch.from_numpy(H).float()
        A1 = torch.from_numpy(A1).float()
        A2 = torch.from_numpy(A2).float()
        V = torch.from_numpy(V).float()

        H, A1, A2, V = H.to(self.device), A1.to(self.device), A2.to(self.device),V.to(self.device)

        return H, A1, A2, V

    def prepare_multi_input(self, list_subgraphs, list_graphs):
        list_inputs = []
        for li, re in zip(list_subgraphs, list_graphs):
            list_inputs.append(self.prepare_single_input(li, re))

        return list_inputs

    def predict_label(self, list_subgraphs, list_graphs):
        list_inputs = self.prepare_multi_input(list_subgraphs, list_graphs)
        input_tensors = self.input_to_tensor(list_inputs)
        results = self.model.test_model(input_tensors)
        return results

    def predict_embedding(self, list_subgraphs, list_graphs):
        list_inputs = self.prepare_multi_input(list_subgraphs, list_graphs)
        input_tensors = self.input_to_tensor(list_inputs)
        results = self.model.get_refined_adjs2(input_tensors)
        return results
        # from scipy.spatial import distance_matrix
        # results = results.cpu().detach().numpy()
        # return [distance_matrix(results[0], results[0])]

def eval_mapping(groundtruth, predict_list, predict_prob):
    acc = []
    MRR = []

    for sgn in groundtruth:
        # Calculate precision
        list_acc = []
        for i in range(1,11):
            if groundtruth[sgn] in predict_list[sgn][:i]:
                list_acc.append(1)
            else:
                list_acc.append(0)

        print(list_acc)
        acc.append(list_acc)

        if groundtruth[sgn] in predict_list[sgn]:
            MRR.append(1 / (predict_list[sgn].index(groundtruth[sgn]) + 1))
        else:
            MRR.append(0)

    acc = np.mean(np.array(acc), axis=0)
    MRR = np.mean(np.array(MRR))
    return np.concatenate([acc, np.array([MRR])])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", "-c", help="checkpoint for gnn", type=str, default="model/best_large_30_20.pt")
    parser.add_argument("--dataset", help="dataset", type=str, default = "tiny")
    parser.add_argument("--num_workers", help="number of workers", type=int, default = os.cpu_count())
    parser.add_argument("--confidence", help="isomorphism threshold", type=float, default = 0.5)
    parser.add_argument("--mapping_threshold", help="mapping threshold", type=float, default = 1e-5)
    parser.add_argument("--ngpu", help="number of gpu", type=int, default = 1)
    parser.add_argument("--batch_size", help="batch_size", type=int, default = 32)
    parser.add_argument("--embedding_dim", help="node embedding dim aka number of distinct node label", type=int, default = 20)
    parser.add_argument("--n_graph_layer", help="number of GNN layer", type=int, default = 4)
    parser.add_argument("--d_graph_layer", help="dimension of GNN layer", type=int, default = 140)
    parser.add_argument("--n_FC_layer", help="number of FC layer", type=int, default = 4)
    parser.add_argument("--d_FC_layer", help="dimension of FC layer", type=int, default = 128)
    parser.add_argument("--dropout_rate", help="dropout_rate", type=float, default = 0.0)
    parser.add_argument("--al_scale", help="attn_loss scale", type=float, default = 1.0)
    parser.add_argument("--tatic", help="tactic of defining number of hops", type=str, default = "static", choices=["static", "continuos", "jump"])
    parser.add_argument("--nhop", help="number of hops", type=int, default = 1)
    parser.add_argument("--data_path", help="path to the data", type=str, default='data_processed')
    parser.add_argument("--result_dir", help="save directory of model parameter", type=str, default = 'results/')
    parser.add_argument("--train_keys", help="train keys", type=str, default='train_keys.pkl')
    parser.add_argument("--test_keys", help="test keys", type=str, default='test_keys.pkl')

    args = parser.parse_args()
    print(args)

    ngpu = args.ngpu
    batch_size = args.batch_size
    data_path = os.path.join(args.data_path, args.dataset)
    args.train_keys = os.path.join(data_path, args.train_keys)
    args.test_keys = os.path.join(data_path, args.test_keys)
    result_dir = os.path.join(args.result_dir, "%s_%s_%d" % (args.dataset, args.tatic, args.nhop))

    if not os.path.isdir(result_dir):
        os.system('mkdir ' + result_dir)

    with open (args.test_keys, 'rb') as fp:
        test_keys = pickle.load(fp)
        # Only use isomorphism subgraphs for mapping testing 
        test_keys = list(filter(lambda x: x.endswith("iso_test"), test_keys))

    print (f'Number of test data: {len(test_keys)}')

    model = gnn(args)
    print ('Number of parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = utils.initialize_model(model, device, load_save_file=args.ckpt)

    test_dataset = BaseDataset(test_keys, data_path, embedding_dim=args.embedding_dim)
    test_dataloader = DataLoader(test_dataset, args.batch_size, \
        shuffle=False, num_workers = args.num_workers, collate_fn=collate_fn)

    # Starting evaluation
    test_true_mapping = []
    test_pred_mapping = []
    list_results = []

    model.eval()
    st_eval = time.time()

    for sample in tqdm(test_dataloader):
        model.zero_grad()
        H, A1, A2, M, S, Y, V, _ = sample 
        H, A1, A2, M, S, Y, V = H.to(device), A1.to(device), A2.to(device),\
                            M.to(device), S.to(device), Y.to(device), V.to(device)
        
        # Test neural network
        pred =  model.get_refined_adjs2((H, A1, A2, V))
        
        # Collect true label and predicted label
        test_true_mapping = M.data.cpu().numpy()
        test_pred_mapping = pred.data.cpu().numpy()

        for mapping_true, mapping_pred in tqdm(zip(test_true_mapping, test_pred_mapping)):
            gt_mapping = {}
            x_coord, y_coord = np.where(mapping_true > 0)
            for x, y in zip(x_coord, y_coord):
                if x < y:
                    gt_mapping[x] = [y] # Subgraph node: Graph node
            
            pred_mapping = defaultdict(lambda: {})
            x_coord, y_coord = np.where(mapping_pred > 0)

            # TODO pred_mapping shoud be sorted by probability

            for x, y in zip(x_coord, y_coord):
                if x < y:
                    if y in pred_mapping[x]:
                        pred_mapping[x][y] = (pred_mapping[x][y] + mapping_pred[x][y])/2
                    else:
                        pred_mapping[x][y] = mapping_pred[x, y] # Subgraph node: Graph node
                else:
                    if x in pred_mapping[y]:
                        pred_mapping[y][x] = (pred_mapping[y][x] + mapping_pred[x][y])/2
                    else:
                        pred_mapping[y][x] = mapping_pred[x, y] # Subgraph node: Graph node

            sorted_predict_mapping = defaultdict(lambda: [])
            sorted_predict_mapping.update({k: [y[0] for y in 
                                        sorted([(n, prob) for n, prob in v.items()], key=lambda x: x[1])]
                                        for k, v in pred_mapping.items()
                                    })

            results = eval_mapping(gt_mapping, sorted_predict_mapping, pred_mapping)
            list_results.append(results)

        if time.time() - st_eval > 10:
            break
            
    end = time.time()

    # test_true_mapping = np.concatenate(np.array(test_true_mapping), 0)
    # test_pred_mapping = np.concatenate(np.array(test_pred_mapping), 0)

    list_results = np.array(list_results)
    avg_results = np.mean(list_results, axis=1)
    print(len(avg_results))
    print(avg_results)

    '''
    # Load subgraph
    subgraphs = utils.read_graphs("data_synthesis/datasets/tiny_30_20/7/iso_subgraphs.lg")
    subgraph = subgraphs[3]
    print("subgraph", subgraph != None)
    utils.plotGraph(subgraph, showLabel=False)
    
    # Load graph
    graphs = utils.read_graphs("data_synthesis/datasets/tiny_30_20/7/source.lg")
    graph = graphs[7]
    print("graph", graph != None)
    # utils.plotGraph(graph, showLabel=True)

    # Load mapping groundtruth
    mapping_gt = utils.read_mapping("data_synthesis/datasets/tiny_30_20/7/iso_subgraphs_mapping.lg")[3]
    print(mapping_gt)

    results = inference_gnn.predict_label([subgraph], [graph])
    print("result", results[0] > args.confidence)

    # if results[0] > args.confidence:
    if True:
        interactions = inference_gnn.predict_embedding([subgraph], [graph])
        # print("interactions", interactions[0])
        n_subgraph_atom = subgraph.number_of_nodes()
        x_coord, y_coord = np.where(interactions[0] > args.mapping_threshold)

        print("Embedding: (subgraph node, graph node)")
        interaction_dict = {}
        for x, y in zip(x_coord, y_coord):
            if x < n_subgraph_atom and y >= n_subgraph_atom:
                interaction_dict[(x, y-n_subgraph_atom)] = interactions[0][x][y]
                # print("(", x, y-n_ligand_atom, ")")

            if x >= n_subgraph_atom and y < n_subgraph_atom and (y, x-n_subgraph_atom) not in interaction_dict:
                interaction_dict[(y, x-n_subgraph_atom)] = interactions[0][x][y]
                # print("(", y, x-n_ligand_atom, ")")

        list_mapping = list(interaction_dict.keys())
        mapping_dict = {}
        for node in subgraph.nodes:
            cnode_mapping = list(map(lambda y: (y[1], interaction_dict[y]), filter(lambda x: x[0] == node, list_mapping)))
            if len(cnode_mapping) == 0:
                mapping_dict[node] = []
                continue

            max_prob = max(cnode_mapping, key = lambda x: x[1])[1]
            mapping_dict[node] = list(map(lambda x: x[0], filter(lambda y: y[1] == max_prob, cnode_mapping)))

        print(mapping_dict)

        node_labels = {n: "" for n in graph.nodes}
        for sgn, list_gn in mapping_dict.items():
            for gn in list_gn:
                if len(node_labels[gn]) == 0:
                    node_labels[gn] = str(sgn)
                else:
                    node_labels[gn] += ",%d" % sgn

        node_colors = {n: "gray" for n in graph.nodes}
        for node, nmaping in node_labels.items():
            if not nmaping:
                if mapping_gt[node] != -1:
                    node_colors[node] = "gold"
                continue

            list_nm = nmaping.split(",")
            for nm in list_nm:
                if mapping_gt[node] == int(nm) and node_colors[node] != "lime":
                    node_colors[node] = "lime"
                
                if mapping_gt[node] != int(nm) and node_colors[node] != "lime":
                    node_colors[node] = "pink"

        for gn, sgn in mapping_gt.items():
            if node_labels[gn] == "" and sgn != -1:
                node_labels[gn] = str(sgn)
                    
        edge_colors = {n: "whitesmoke" for n in graph.edges}
        for edge in graph.edges:
            n1, n2 = edge
            n1_sgs, n2_sgs = node_labels[n1], node_labels[n2] # map node from graph to node in subgraph

            if node_colors[n1] == "gray" or node_colors[n2] == "gray":
                continue

            # Check wheather a link between n1, n2 in subgraph
            total_pair = len(n1_sgs.split(",")) * len(n2_sgs.split(","))
            count_pair = 0
            for n1_sg in n1_sgs.split(","):
                n1_sg = int(n1_sg)
                for n2_sg in n2_sgs.split(","):
                    n2_sg = int(n2_sg)
                    if (n1_sg, n2_sg) not in subgraph.edges and (n2_sg, n1_sg) not in subgraph.edges:
                        count_pair += 1

            if count_pair != total_pair:
                if node_colors[n1] == "lime" and node_colors[n2] == "lime":
                    edge_colors[edge] = "black"
                elif node_colors[n1] == "gold" or node_colors[n2] == "gold":
                    edge_colors[edge] = "goldenrod"
                elif node_colors[n1] == "pink" or node_colors[n2] == "pink":
                    edge_colors[edge] = "palevioletred"
            else:
                if node_colors[n1] == "pink" or node_colors[n2] == "pink":
                    edge_colors[edge] = "palevioletred"

        utils.plotGraph(graph, nodeLabels=node_labels, 
                        nodeColors=list(node_colors.values()), 
                        edgeColors=list(edge_colors.values()))

        with open("results/mapping_%s.csv" % datetime.now().strftime("%Y-%m-%dT%H-%M-%S"), "w", encoding="utf8") as f:
            f.write("subgraph_node,graph_node,score\n")
            for key, value in interaction_dict.items():
                f.write("{:d},{:d},{:.3e}\n".format(key[0], key[1], value))

    '''