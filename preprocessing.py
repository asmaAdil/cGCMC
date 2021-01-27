from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.sparse as sp
import pickle as pkl
import os
import h5py
import pandas as pd
import pdb
from data_utils import load_data, map_data, download_dataset


def normalize_features(feat):

    sum= feat.sum(1)  # (121 x 1)
    sum_flat=np.array(sum.flatten())  # (1 x 121)

    degree = np.asarray(feat.sum(1)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree[degree == 0.] = np.inf

    degree_inv = 1. / degree
    degree_inv_mat = sp.diags([degree_inv], [0]) #degree (121, 121)

    feat_norm = degree_inv_mat.dot(feat) #degree (121 x 121) x (121 x 7)

    if feat_norm.nnz == 0:
        print('ERROR: normalized adjacency matrix has only zero entries!!!!!')
        exit

    return feat_norm


def normalize_edge_features_3D(feat):
    sum= feat.sum(2)  # (121 x 1232 x 49) => (121 x1232)
    degree=sum.reshape(-1)

    # set zeros to inf to avoid dividing by zero
    degree[degree == 0.] = np.inf

    degree_inv = 1. / degree
    degree_inv_mat = sp.diags([degree_inv], [0]) #(149072, 149072)

    feat_r=np.reshape(feat,((feat.shape[0]*feat.shape[1]), feat.shape[2]))   #(149072, 49)

    feat_norm = degree_inv_mat.dot(feat_r) #(149072, 149072)  x (149072, 49) = (149072, 49)

    if np.nonzero(feat_norm) == 0:
        print('ERROR: normalized adjacency matrix has only zero entries!!!!!')
        exit
    feat_norm=feat_norm.reshape(feat.shape[0],feat.shape[1],feat.shape[2])

    return feat_norm



def normalize_edge_features_2D(feat):

    sum= feat.sum(1)
    degree=sum.reshape(-1)

    # set zeros to inf to avoid dividing by zero
    degree[degree == 0.] = np.inf

    degree_inv = 1. / degree
    degree_inv_mat = sp.diags([degree_inv], [0])

    feat_norm = degree_inv_mat.dot(feat)

    if np.nonzero(feat_norm) == 0:
        print('ERROR: normalized adjacency matrix has only zero entries!!!!!')
        exit

    return feat_norm

def contextNormalize(feat):
    n_context=feat.shape[2]
    feat=np.sum(feat, axis=2)
    feat=np.divide(feat,n_context)
    print(f"feat {feat.shape}")
    feat_t=np.transpose(feat)
    feat= sp.csr_matrix(feat)
    feat_t = sp.csr_matrix(feat_t)
    return feat, feat_t


def normalize_edge_features_3Dto_2D(feat):

    prob_r = [0.2, 0.3, 0.5, 0.7, 0.8]
    i=0
    adj_tot = [np.sum(adj,axis=2) for adj in feat]
    adjacencies_prioritize = adj_tot
    for adj in adjacencies_prioritize:
        adj = adj * prob_r[i]
        i += 1
    adj_sp = [sp.csr_matrix(adj) for adj in adjacencies_prioritize]
    adj_sp = globally_normalize_bipartite_adjacency(adj_sp)

    return adj_sp

def load_matlab_file(path_file, name_field):
    """
    load '.mat' files
    inputs:
        path_file, string containing the file path
        name_field, string containig the field name (default='shape')
    warning:
        '.mat' files should be saved in the '-v7.3' format
    """
    db = h5py.File(path_file, 'r')
    ds = db[name_field]
    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['data'])
            ir = np.asarray(ds['ir'])
            jc = np.asarray(ds['jc'])
            out = sp.csc_matrix((data, ir, jc)).astype(np.float32)
    except AttributeError:
        # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
        out = np.asarray(ds).astype(np.float32).T

    db.close()

    return out


def preprocess_user_item_features(u_features, v_features):
    """
    Creates one big feature matrix out of user features and item features.
    Stacks item features under the user features.
    """

    zero_csr_u = sp.csr_matrix((u_features.shape[0], v_features.shape[1]), dtype=u_features.dtype)   #121 x 1232
    zero_csr_v = sp.csr_matrix((v_features.shape[0], u_features.shape[1]), dtype=v_features.dtype)  # 1232 x 121

    u_features = sp.hstack([u_features, zero_csr_u], format='csr')   # 121 x 121  stack 121 x 1232= 121 x [121 + 1232]

    v_features = sp.hstack([zero_csr_v, v_features], format='csr')  # 1232 x 121  stack 1232 x 1232= 1232 x [121 + 1232]

    return u_features, v_features


def globally_normalize_bipartite_adjacency(adjacencies, verbose=False, symmetric=True):
    """ Globally Normalizes set of bipartite adjacency matrices """

    #a=isinstance(adjacencies,list) #true
    if verbose:
        print('Symmetrically normalizing bipartite adj')
    # degree_u and degree_v are row and column sums of adj+I


    adj_tot = np.sum(adj for adj in adjacencies)
    degree_u = np.asarray(adj_tot.sum(1)).flatten()
    degree_v = np.asarray(adj_tot.sum(0)).flatten()
    print(f"degree_u {degree_u.shape} degree_v {degree_v.shape}")

    # set zeros to inf to avoid dividing by zero
    degree_u[degree_u == 0.] = np.inf
    degree_v[degree_v == 0.] = np.inf

    degree_u_inv_sqrt = 1. / np.sqrt(degree_u)  # 1 /sqroot degree of u
    degree_v_inv_sqrt = 1. / np.sqrt(degree_v)  # 1 /sqroot degree of v
    degree_u_inv_sqrt_mat = sp.diags([degree_u_inv_sqrt], [0])
    degree_v_inv_sqrt_mat = sp.diags([degree_v_inv_sqrt], [0])

    print(f" A : degree_u_inv_sqrt_mat {degree_u_inv_sqrt_mat.shape} degree_u_inv_sqrt_mat {degree_u_inv_sqrt_mat.shape}")
    degree_u_inv = degree_u_inv_sqrt_mat.dot(degree_u_inv_sqrt_mat)
    print(f"B: degree_u_inv {degree_u_inv.shape} ")

    if symmetric:
        print(f" C : degree_u_inv_sqrt_mat {degree_u_inv_sqrt_mat.shape} adj {adjacencies[0].shape} degree_v_inv_sqrt_mat {degree_v_inv_sqrt_mat.shape}")
        #print("yes sym") called for ml _100k
        adj_norm = [degree_u_inv_sqrt_mat.dot(adj).dot(degree_v_inv_sqrt_mat) for adj in adjacencies]

    else:
        adj_norm = [degree_u_inv.dot(adj) for adj in adjacencies]
    print(f"D adj_norm {adj_norm[0].shape} ")
    return adj_norm

def globally_normalize_tripartite_adjacency_matrix(adjacencies, verbose=False, symmetric=True):
        """ Globally Normalizes set of bipartite adjacency matrices """

        # a=isinstance(adjacencies,list) #true
        if verbose:
            print('Symmetrically normalizing bipartite adj')
        # degree_u and degree_v are row and column sums of adj+I
        adjacencies_uv_c= [np.reshape(adj,(adj.shape[0]*adj.shape[1], adj.shape[2])) for adj in  adjacencies]

        adj_tot = np.sum(adj for adj in adjacencies)
        adj_tot_uv_c=np.reshape(adj_tot, (adj_tot.shape[0]*adj_tot.shape[1], adj_tot.shape[2]))

        adj_tot_t=np.transpose(adj_tot, (1,0,2))
        adj_tot_vu_c = np.reshape(adj_tot_t, (adj_tot_t.shape[0] * adj_tot_t.shape[1], adj_tot_t.shape[2]))

        degree_uv_c = np.asarray(adj_tot_uv_c.sum(1)).flatten()
        degree_vu_c = np.asarray(adj_tot_vu_c.sum(1)).flatten()

        # set zeros to inf to avoid dividing by zero
        degree_uv_c[degree_uv_c == 0.] = np.inf
        degree_vu_c[degree_vu_c == 0.] = np.inf

        degree_uv_inv_sqrt = 1. / np.sqrt(degree_uv_c)  # 1 /sqroot degree of u
        degree_vu_inv_sqrt = 1. / np.sqrt(degree_vu_c)  # 1 /sqroot degree of v
        degree_uv_inv_sqrt_mat = sp.diags([degree_uv_inv_sqrt], [0])
        degree_vu_inv_sqrt_mat = sp.diags([degree_vu_inv_sqrt], [0])

        degree_uv_inv = degree_uv_inv_sqrt_mat.dot(degree_uv_inv_sqrt_mat)
        degree_vu_inv = degree_vu_inv_sqrt_mat.dot(degree_vu_inv_sqrt_mat)

        if symmetric:
            # print("yes sym") called for ml _100k
            adj_norm_uv_c = [degree_uv_inv_sqrt_mat.dot(adj) for adj in adjacencies_uv_c]
            adj_norm_u_v_c=[np.reshape(adj, (adjacencies[0].shape[0],adjacencies[0].shape[1], adjacencies[0].shape[2])) for adj in adj_norm_uv_c]
            adj_norm_v_u_c = [np.transpose(adj, (1, 0, 2)) for adj in adj_norm_u_v_c]
            adj_norm_vu_c=[np.reshape(adj, (adj.shape[0]*adj.shape[1], adj.shape[2])) for adj in adj_norm_v_u_c]
            adj_norm_vu_c = [degree_vu_inv_sqrt_mat.dot(adj) for adj in adj_norm_vu_c]
            adj_norm = [np.reshape(adj,(adjacencies[0].shape[0], adjacencies[0].shape[1], adjacencies[0].shape[2])) for adj in adj_norm_vu_c]

        else:
            adj_norm = [degree_uv_inv.dot(adj) for adj in adjacencies]

        print(f"adj_normc {adj_norm[0].shape}")

        return adj_norm

def user_context_adjacency(adjacencies):
     """ Find importance of context for users
     giving high probability to context with rating 5
     """
     adj_tot = np.sum(adj for adj in adjacencies)
     deg_u=np.sum(adj_tot, axis = 1)
     deg_u = np.sum(deg_u, axis=1)


     # set zeros to inf to avoid dividing by zero
     deg_u[deg_u == 0.] = np.inf
     degree_u_inv_sqrt = 1. / np.sqrt(deg_u)

     degree_u_inv_sqrt_mat = sp.diags([degree_u_inv_sqrt], [0])
     adju_c=[np.sum(adj, axis=1) for adj in adjacencies]

     adju_c_norm = [degree_u_inv_sqrt_mat.dot(adj) for adj in adju_c]
     #normalize this matrix by divisding squareroot of degtree

     #print(f"degree_u_inv_sqrt_mat shape {degree_u_inv_sqrt_mat.shape}  {degree_u_inv_sqrt_mat}")
     prob_r=[0.2,0.3,0.5,0.7,0.8]
     i=0
     adjacencies_prioritize=adju_c_norm
     for adj in adjacencies_prioritize:
         adj= adj * prob_r[i]
         i+=1

     adju_c_imp=np.sum(adj for adj in adjacencies_prioritize)

     #adjacencies_temp_tot = np.sum(adj for adj in adjacencies_temp)
     return adju_c_imp


def item_context_adjacency(adjacencies):
    """ Find importance of context for users
    giving high probability to context with rating 5
    """
    adj_tot = np.sum(adj for adj in adjacencies)
    deg_v = np.sum(adj_tot, axis=1)
    deg_v = np.sum(deg_v, axis=1)

    # set zeros to inf to avoid dividing by zero
    deg_v[deg_v == 0.] = np.inf
    degree_v_inv_sqrt = 1. / np.sqrt(deg_v)

    degree_v_inv_sqrt_mat = sp.diags([degree_v_inv_sqrt], [0])
    adjv_c = [np.sum(adj, axis=1) for adj in adjacencies]

    adjv_c_norm = [degree_v_inv_sqrt_mat.dot(adj) for adj in adjv_c]
    # normalize this matrix by divisding squareroot of degtree

    prob_r = [0.2, 0.3, 0.5, 0.7, 0.8]
    i = 0
    adjacencies_prioritize = adjv_c_norm
    for adj in adjacencies_prioritize:
        adj = adj * prob_r[i]
        i += 1
    adjv_c_imp = np.sum(adj for adj in adjacencies_prioritize)

    # adjacencies_temp_tot = np.sum(adj for adj in adjacencies_temp)
    return adjv_c_imp


def user_context_fromedge(adjacency):
    """ Find importance of context for users
    giving high probability to context with rating 5
    """

    deg_u = np.sum(adjacency, axis=1)
    deg_u = np.sum(deg_u, axis=1)
    print(f"degree_u {deg_u.shape}")

    # set zeros to inf to avoid dividing by zero
    deg_u[deg_u == 0.] = np.inf
    degree_u_inv_sqrt = 1. / np.sqrt(deg_u)

    degree_u_inv_sqrt_mat = sp.diags([degree_u_inv_sqrt], [0])
    adju_c = np.sum(adjacency, axis=1)

    adju_c_norm = degree_u_inv_sqrt_mat.dot(adju_c)
    # normalize this matrix by divisding squareroot of degtree

    return adju_c_norm

def item_context_fromedge(adjacency):
    """ Find importance of context for users
    giving high probability to context with rating 5
    """
    deg_v = np.sum(adjacency, axis=1)
    deg_v = np.sum(deg_v, axis=1)


    # set zeros to inf to avoid dividing by zero
    deg_v[deg_v == 0.] = np.inf
    degree_v_inv_sqrt = 1. / np.sqrt(deg_v)

    degree_v_inv_sqrt_mat = sp.diags([degree_v_inv_sqrt], [0])
    adjv_c = np.sum(adjacency, axis=1)

    adjv_c_norm = degree_v_inv_sqrt_mat.dot(adjv_c)

    # adjacencies_temp_tot = np.sum(adj for adj in adjacencies_temp)
    return adjv_c_norm


def sparse_to_tuple(sparse_mx):
    """ change of format for sparse matrix. This format is used
    for the feed_dict where sparse matrices need to be linked to placeholders
    representing sparse matrices. """

    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def create_trainvaltest_split(dataset, seed=1234, testing=False, datasplit_path=None, datasplit_from_file=False,
                              verbose=True, rating_map=None, post_rating_map=None, ratio=1.0):
    """
    Splits data set into train/val/test sets from full bipartite adjacency matrix. Shuffling of dataset is done in
    load_data function.
    For each split computes 1-of-num_classes labels. Also computes training
    adjacency matrix.
    """

    if datasplit_from_file and os.path.isfile(datasplit_path):
        print('Reading dataset splits from file...')
        with open(datasplit_path, 'rb') as f:
            num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features = pkl.load(f)

        if verbose:
            print('Number of users = %d' % num_users)
            print('Number of items = %d' % num_items)
            print('Number of links = %d' % ratings.shape[0])
            print('Fraction of positive links = %.4f' % (float(ratings.shape[0]) / (num_users * num_items),))

    else:
        print(f"I am preprocessing {dataset} ")
        num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features, edge_feactures = load_data(dataset, seed=seed,
                                                                                            verbose=verbose)

        with open(datasplit_path, 'wb') as f:
            pkl.dump([num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features], f)

    if rating_map is not None:
        for i, x in enumerate(ratings):
            ratings[i] = rating_map[x]

    neutral_rating = -1

    rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}

    labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
    labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings])
    labels = labels.reshape([-1])

    # number of test and validation edges
    num_test = int(np.ceil(ratings.shape[0] * 0.1))
    if dataset == 'ml_100k':
        num_val = int(np.ceil(ratings.shape[0] * 0.9 * 0.05))
    else:
        num_val = int(np.ceil(ratings.shape[0] * 0.9 * 0.05))

    num_train = ratings.shape[0] - num_val - num_test

    pairs_nonzero = np.array([[u, v] for u, v in zip(u_nodes, v_nodes)])

    idx_nonzero = np.array([u * num_items + v for u, v in pairs_nonzero])

    train_idx = idx_nonzero[0:int(num_train * ratio)]
    val_idx = idx_nonzero[num_train:num_train + num_val]
    test_idx = idx_nonzero[num_train + num_val:]

    train_pairs_idx = pairs_nonzero[0:int(num_train * ratio)]
    val_pairs_idx = pairs_nonzero[num_train:num_train + num_val]
    test_pairs_idx = pairs_nonzero[num_train + num_val:]

    u_test_idx, v_test_idx = test_pairs_idx.transpose()
    u_val_idx, v_val_idx = val_pairs_idx.transpose()
    u_train_idx, v_train_idx = train_pairs_idx.transpose()

    # create labels
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    if testing:
        u_train_idx = np.hstack([u_train_idx, u_val_idx])
        v_train_idx = np.hstack([v_train_idx, v_val_idx])
        train_labels = np.hstack([train_labels, val_labels])
        # for adjacency matrix construction
        train_idx = np.hstack([train_idx, val_idx])

    class_values = np.sort(np.unique(ratings))

    # make training adjacency matrix
    rating_mx_train = np.zeros(num_users * num_items, dtype=np.float32)
    if post_rating_map is None:
        rating_mx_train[train_idx] = labels[train_idx].astype(np.float32) + 1.
    else:
        rating_mx_train[train_idx] = np.array([post_rating_map[r] for r in class_values[labels[train_idx]]]) + 1.
    rating_mx_train = sp.csr_matrix(rating_mx_train.reshape(num_users, num_items))

    return u_features, v_features, rating_mx_train, train_labels, u_train_idx, v_train_idx, \
           val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx, v_test_idx, class_values

def create_trainvaltest_split_Context( LDcall=False, dataset='LDOS', seed=1234, testing=False, datasplit_path=None, datasplit_from_file=False,
                              verbose=True, rating_map=None, post_rating_map=None, ratio=1.0):
     """
     Splits data set into train/val/test sets from full bipartite adjacency matrix. Shuffling of dataset is done in
     load_data function.
     For each split computes 1-of-num_classes labels. Also computes training
     adjacency matrix.
     """
     edge_f_mx_train=None
     u_features=v_features=rating_mx_train=train_labels=u_train_idx=v_train_idx=None
     val_labels=u_val_idx=v_val_idx=test_labels=u_test_idx=v_test_idx=class_values=None
     if datasplit_from_file and os.path.isfile(datasplit_path):
        print('Reading dataset splits from file...')
        with open(datasplit_path, 'rb') as f:
            num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features, edge_features = pkl.load(f)

        if verbose:
            print('Number of users = %d' % num_users)
            print('Number of items = %d' % num_items)
            print('Number of links = %d' % ratings.shape[0])
            print('Fraction of positive links = %.4f' % (float(ratings.shape[0]) / (num_users * num_items),))

     else:
        print("split() in processig calling load() in data_utils")
        num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features, edge_features = load_data(dataset, seed=seed,
                                                                                            verbose=verbose)


        with open(datasplit_path, 'wb') as f:
            pkl.dump([num_users, num_items, u_nodes, v_nodes, ratings, u_features, v_features, edge_features], f)

     if rating_map is not None:
        for i, x in enumerate(ratings):
            ratings[i] = rating_map[x]


     neutral_rating = -1

     rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}  #{0:1, 1:2, 2:3, 3:4, 4:5}
     labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
     labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings]) #(121, 1232)
     a=np.array([rating_dict[r] for r in ratings]) #(121, 1232)
     labels = labels.reshape([-1])  #(149072,)

     edge_f = np.full((num_users, num_items, edge_features.shape[2]),0, dtype=np.float32)

     for u,v in zip(u_nodes,v_nodes):
        edge_f[u, v, :]=edge_features[u, v, :]

     edge_f=np.reshape(edge_f,((edge_f.shape[0]*edge_f.shape[1]),edge_f.shape[2]))
     ind = np.array(np.where(edge_f != 0)).T  # (u v c) #non zero indices

     #edge_f[u_nodes, v_nodes] = np.array(np.array([edge_features_d[i] for i in range(2096)]))  # (121, 1232)


     # number of test and validation edges
     num_test = int(np.ceil(ratings.shape[0] * 0.1))

     if dataset == 'ml_100k':
        num_val = int(np.ceil(ratings.shape[0] * 0.9 * 0.05))
     else:
        num_val = int(np.ceil(ratings.shape[0] * 0.9 * 0.05))

     num_train = ratings.shape[0] - num_val - num_test
     print("*****************************Splitting()****************************************")
     print(f"num_test {num_test} num_val {num_val} num_train {num_train} ")

     pairs_nonzero = np.array([[u, v] for u, v in zip(u_nodes, v_nodes)])
     idx_nonzero = np.array([u * num_items + v for u, v in pairs_nonzero])

     train_idx = idx_nonzero[0:int(num_train * ratio)]
     val_idx = idx_nonzero[num_train:num_train + num_val]
     test_idx = idx_nonzero[num_train + num_val:]

     #train_idx 0 - 1962.0 val_idx 1962 - 2066  test_idx 2066

     train_pairs_idx = pairs_nonzero[0:int(num_train * ratio)]
     val_pairs_idx = pairs_nonzero[num_train:num_train + num_val]
     test_pairs_idx = pairs_nonzero[num_train + num_val:]


     u_test_idx, v_test_idx = test_pairs_idx.transpose()
     u_val_idx, v_val_idx = val_pairs_idx.transpose()
     u_train_idx, v_train_idx = train_pairs_idx.transpose()

     # create labels
     train_labels = labels[train_idx]
     val_labels = labels[val_idx]
     test_labels = labels[test_idx]

     #print(f"train-val-test labels {train_labels.shape} - {val_idx.shape} - {test_idx.shape}")
     #ndArray[start_index: end_index ,  :]

     train_edge_f=edge_f[train_idx ,  :]
     val_edge_f=edge_f[val_idx, : ]
     test_edge_f=edge_f[test_idx, : ]

     testing=False

     if testing:
        u_train_idx = np.hstack([u_train_idx, u_val_idx])
        v_train_idx = np.hstack([v_train_idx, v_val_idx])

        train_labels = np.hstack([train_labels, val_labels])
        train_edge_f = np.concatenate([train_edge_f, val_edge_f]) #train_labels (2066,) train_edge_f(2066, 49)

        # for adjacency matrix construction
        train_idx = np.hstack([train_idx, val_idx])

     class_values = np.sort(np.unique(ratings))

     # make training adjacency matrix
     rating_mx_train = np.zeros(num_users * num_items, dtype=np.float32)
     edge_f_mx_train = np.zeros([num_users*num_items,  edge_features.shape[2]], dtype=np.float32)

     edge_f_mx_test=np.zeros([num_users*num_items,  edge_features.shape[2]], dtype=np.float32)
     edge_f_mx_val = np.zeros([num_users * num_items, edge_features.shape[2]], dtype=np.float32)

     if post_rating_map is None:
        rating_mx_train[train_idx] = labels[train_idx].astype(np.float32) + 1.
        edge_f_mx_train[train_idx, : ]=edge_f[train_idx, :].astype(np.float32)

        edge_f_mx_test[test_idx, :] = edge_f[test_idx, :].astype(np.float32)
        edge_f_mx_val[val_idx, :] = edge_f[val_idx, :].astype(np.float32)

     else:
        rating_mx_train[train_idx] = np.array([post_rating_map[r] for r in class_values[labels[train_idx]]]) + 1.

     rating_mx_train = sp.csr_matrix(rating_mx_train.reshape(num_users, num_items))
     edge_f_mx_train=edge_f_mx_train.reshape(num_users,num_items,edge_features.shape[2])

     edge_f_mx_test = edge_f_mx_test.reshape(num_users, num_items, edge_features.shape[2])
     edge_f_mx_val = edge_f_mx_val.reshape(num_users, num_items, edge_features.shape[2])

     """
     train_edge_f = sp.csr_matrix(train_edge_f)
     test_edge_f = sp.csr_matrix(test_edge_f)
     val_edge_f = sp.csr_matrix(val_edge_f)
     """

     print(f"***************************************************************************")
     if (u_features is not None) and (v_features is not None):
        print(f"u_features-sp.csr-matrix {u_features.shape}  v_features-sp.csr-matrix {v_features.shape}")
     print(f" Train: rating_mx_trainsp.csr-matrix {rating_mx_train.shape} edge_f_mx_train-nparray {edge_f_mx_train.shape} train_edge_f-nparray {train_edge_f.shape} train_labels-1DT {train_labels.shape} u_train_idx-1DT {u_train_idx.shape} v_train_idx-1DT {v_train_idx.shape}")
     print(f"Validation : val_labels-1DT {val_labels.shape} edge_f_mx_val-nparray {edge_f_mx_val.shape} val_edge_f-nparray {val_edge_f.shape} u_val_idx-1D {u_val_idx.shape} v_val_idx-1DT {v_val_idx.shape}")
     print(f"Testing: test_labels-1DT {test_labels.shape} edge_f_mx_test-nparray {edge_f_mx_test.shape}  test_edge_f-nparray {test_edge_f.shape} u_test_idx-1DT {u_test_idx.shape}  v_test_idx-1Dt {v_test_idx.shape}")
     print(f" class_values {class_values}")
     print(f"******************************************************************************")

     return  u_features, v_features, rating_mx_train, edge_f_mx_train, train_edge_f, train_labels, u_train_idx, v_train_idx, \
           val_labels, edge_f_mx_val, val_edge_f, u_val_idx, v_val_idx, test_labels, edge_f_mx_test, test_edge_f, u_test_idx, v_test_idx, class_values

def load_data_monti(dataset, testing=False, rating_map=None, post_rating_map=None):
    """
    Loads data from Monti et al. paper.
    if rating_map is given, apply this map to the original rating matrix
    if post_rating_map is given, apply this map to the processed rating_mx_train without affecting the labels
    """

    path_dataset = 'raw_data/' + dataset + '/training_test_dataset.mat'

    M = load_matlab_file(path_dataset, 'M')
    if rating_map is not None:
        M[np.where(M)] = [rating_map[x] for x in M[np.where(M)]]

    Otraining = load_matlab_file(path_dataset, 'Otraining')
    Otest = load_matlab_file(path_dataset, 'Otest')

    num_users = M.shape[0]
    num_items = M.shape[1]

    if dataset == 'flixster':
        Wrow = load_matlab_file(path_dataset, 'W_users')
        Wcol = load_matlab_file(path_dataset, 'W_movies')
        u_features = Wrow
        v_features = Wcol
    elif dataset == 'douban':
        Wrow = load_matlab_file(path_dataset, 'W_users')
        u_features = Wrow
        v_features = np.eye(num_items)
    elif dataset == 'yahoo_music':
        Wcol = load_matlab_file(path_dataset, 'W_tracks')
        u_features = np.eye(num_users)
        v_features = Wcol

    u_nodes_ratings = np.where(M)[0]
    v_nodes_ratings = np.where(M)[1]
    ratings = M[np.where(M)]

    u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int32)
    ratings = ratings.astype(np.float64)

    u_nodes = u_nodes_ratings
    v_nodes = v_nodes_ratings

    print('number of users = ', len(set(u_nodes)))
    print('number of item = ', len(set(v_nodes)))

    neutral_rating = -1  # int(np.ceil(np.float(num_classes)/2.)) - 1

    # assumes that ratings_train contains at least one example of every rating type
    rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}

    labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
    labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings])

    for i in range(len(u_nodes)):
        assert (labels[u_nodes[i], v_nodes[i]] == rating_dict[ratings[i]])

    labels = labels.reshape([-1])

    # number of test and validation edges

    num_train = np.where(Otraining)[0].shape[0]
    num_test = np.where(Otest)[0].shape[0]
    num_val = int(np.ceil(num_train * 0.2))
    num_train = num_train - num_val

    pairs_nonzero_train = np.array([[u, v] for u, v in zip(np.where(Otraining)[0], np.where(Otraining)[1])])
    idx_nonzero_train = np.array([u * num_items + v for u, v in pairs_nonzero_train])

    pairs_nonzero_test = np.array([[u, v] for u, v in zip(np.where(Otest)[0], np.where(Otest)[1])])
    idx_nonzero_test = np.array([u * num_items + v for u, v in pairs_nonzero_test])

    # Internally shuffle training set (before splitting off validation set)
    rand_idx = list(range(len(idx_nonzero_train)))
    np.random.seed(42)
    np.random.shuffle(rand_idx)
    idx_nonzero_train = idx_nonzero_train[rand_idx]
    pairs_nonzero_train = pairs_nonzero_train[rand_idx]

    idx_nonzero = np.concatenate([idx_nonzero_train, idx_nonzero_test], axis=0)
    pairs_nonzero = np.concatenate([pairs_nonzero_train, pairs_nonzero_test], axis=0)

    val_idx = idx_nonzero[0:num_val]
    train_idx = idx_nonzero[num_val:num_train + num_val]
    test_idx = idx_nonzero[num_train + num_val:]

    assert (len(test_idx) == num_test)

    val_pairs_idx = pairs_nonzero[0:num_val]
    train_pairs_idx = pairs_nonzero[num_val:num_train + num_val]
    test_pairs_idx = pairs_nonzero[num_train + num_val:]

    u_test_idx, v_test_idx = test_pairs_idx.transpose()
    u_val_idx, v_val_idx = val_pairs_idx.transpose()
    u_train_idx, v_train_idx = train_pairs_idx.transpose()

    # create labels
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    if testing:
        u_train_idx = np.hstack([u_train_idx, u_val_idx])
        v_train_idx = np.hstack([v_train_idx, v_val_idx])
        train_labels = np.hstack([train_labels, val_labels])
        # for adjacency matrix construction
        train_idx = np.hstack([train_idx, val_idx])

    class_values = np.sort(np.unique(ratings))

    # make training adjacency matrix
    rating_mx_train = np.zeros(num_users * num_items, dtype=np.float32)
    '''Note here rating matrix elements' values + 1 !!!'''
    if post_rating_map is None:
        rating_mx_train[train_idx] = labels[train_idx].astype(np.float32) + 1.
    else:
        rating_mx_train[train_idx] = np.array([post_rating_map[r] for r in class_values[labels[train_idx]]]) + 1.

    rating_mx_train = sp.csr_matrix(rating_mx_train.reshape(num_users, num_items))

    if u_features is not None:
        u_features = sp.csr_matrix(u_features)
        print("User features shape: " + str(u_features.shape))

    if v_features is not None:
        v_features = sp.csr_matrix(v_features)
        print("Item features shape: " + str(v_features.shape))

    return u_features, v_features, rating_mx_train, train_labels, u_train_idx, v_train_idx, \
           val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx, v_test_idx, class_values


def load_official_trainvaltest_split(dataset, testing=False, rating_map=None, post_rating_map=None, ratio=1.0):
    """
    Loads official train/test split and uses 10% of training samples for validaiton
    For each split computes 1-of-num_classes labels. Also computes training
    adjacency matrix. Assumes flattening happens everywhere in row-major fashion.
    """

    sep = '\t'

    # Check if files exist and download otherwise
    files = ['/u1.base', '/u1.test', '/u.item', '/u.user']
    fname = dataset
    data_dir = 'raw_data/' + fname

    download_dataset(fname, files, data_dir)
    print(f"load_official_trainvaltest_split {fname}")

    dtypes = {
        'u_nodes': np.int32, 'v_nodes': np.int32,
        'ratings': np.float32, 'timestamp': np.float64}

    filename_train = 'raw_data/' + dataset + '/u1.base'
    filename_test = 'raw_data/' + dataset + '/u1.test'

    data_train = pd.read_csv(
        filename_train, sep=sep, header=None,
        names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'], dtype=dtypes)

    data_test = pd.read_csv(
        filename_test, sep=sep, header=None,
        names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'], dtype=dtypes)

    data_array_train = data_train.values.tolist()
    data_array_train = np.array(data_array_train)
    data_array_test = data_test.values.tolist()
    data_array_test = np.array(data_array_test)

    if ratio < 1.0:
        data_array_train = data_array_train[data_array_train[:, -1].argsort()[:int(ratio * len(data_array_train))]]

    data_array = np.concatenate([data_array_train, data_array_test], axis=0)

    u_nodes_ratings = data_array[:, 0].astype(dtypes['u_nodes'])
    v_nodes_ratings = data_array[:, 1].astype(dtypes['v_nodes'])
    ratings = data_array[:, 2].astype(dtypes['ratings'])
    if rating_map is not None:
        for i, x in enumerate(ratings):
            ratings[i] = rating_map[x]

    u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
    v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)

    u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int32)
    ratings = ratings.astype(np.float64)

    u_nodes = u_nodes_ratings
    v_nodes = v_nodes_ratings

    neutral_rating = -1  # int(np.ceil(np.float(num_classes)/2.)) - 1

    # assumes that ratings_train contains at least one example of every rating type
    rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}

    labels = np.full((num_users, num_items), neutral_rating, dtype=np.int32)
    labels[u_nodes, v_nodes] = np.array([rating_dict[r] for r in ratings])

    for i in range(len(u_nodes)):
        assert (labels[u_nodes[i], v_nodes[i]] == rating_dict[ratings[i]])

    labels = labels.reshape([-1])

    # number of test and validation edges, see cf-nade code

    num_train = data_array_train.shape[0]
    num_test = data_array_test.shape[0]
    num_val = int(np.ceil(num_train * 0.2))
    num_train = num_train - num_val

    pairs_nonzero = np.array([[u, v] for u, v in zip(u_nodes, v_nodes)])
    idx_nonzero = np.array([u * num_items + v for u, v in pairs_nonzero])

    for i in range(len(ratings)):
        assert (labels[idx_nonzero[i]] == rating_dict[ratings[i]])

    idx_nonzero_train = idx_nonzero[0:num_train + num_val]
    idx_nonzero_test = idx_nonzero[num_train + num_val:]

    pairs_nonzero_train = pairs_nonzero[0:num_train + num_val]
    pairs_nonzero_test = pairs_nonzero[num_train + num_val:]

    # Internally shuffle training set (before splitting off validation set)
    rand_idx = list(range(len(idx_nonzero_train)))
    np.random.seed(42)
    np.random.shuffle(rand_idx)
    idx_nonzero_train = idx_nonzero_train[rand_idx]
    pairs_nonzero_train = pairs_nonzero_train[rand_idx]

    idx_nonzero = np.concatenate([idx_nonzero_train, idx_nonzero_test], axis=0)
    pairs_nonzero = np.concatenate([pairs_nonzero_train, pairs_nonzero_test], axis=0)

    val_idx = idx_nonzero[0:num_val]
    train_idx = idx_nonzero[num_val:num_train + num_val]
    test_idx = idx_nonzero[num_train + num_val:]

    assert (len(test_idx) == num_test)

    val_pairs_idx = pairs_nonzero[0:num_val]
    train_pairs_idx = pairs_nonzero[num_val:num_train + num_val]
    test_pairs_idx = pairs_nonzero[num_train + num_val:]

    u_test_idx, v_test_idx = test_pairs_idx.transpose()
    u_val_idx, v_val_idx = val_pairs_idx.transpose()
    u_train_idx, v_train_idx = train_pairs_idx.transpose()

    # create labels
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    if testing:
        u_train_idx = np.hstack([u_train_idx, u_val_idx])
        v_train_idx = np.hstack([v_train_idx, v_val_idx])
        train_labels = np.hstack([train_labels, val_labels])
        # for adjacency matrix construction
        train_idx = np.hstack([train_idx, val_idx])

    class_values = np.sort(np.unique(ratings))

    # make training adjacency matrix
    rating_mx_train = np.zeros(num_users * num_items, dtype=np.float32)
    if post_rating_map is None:
        rating_mx_train[train_idx] = labels[train_idx].astype(np.float32) + 1.
    else:
        rating_mx_train[train_idx] = np.array([post_rating_map[r] for r in class_values[labels[train_idx]]]) + 1.
    rating_mx_train = sp.csr_matrix(rating_mx_train.reshape(num_users, num_items))

    if dataset == 'ml_100k':

        # movie features (genres)
        sep = r'|'
        movie_file = 'raw_data/' + dataset + '/u.item'
        movie_headers = ['movie id', 'movie title', 'release date', 'video release date',
                         'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
                         'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                         'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                         'Thriller', 'War', 'Western']
        movie_df = pd.read_csv(movie_file, sep=sep, header=None,
                               names=movie_headers, engine='python')

        genre_headers = movie_df.columns.values[6:]
        num_genres = genre_headers.shape[0]

        v_features = np.zeros((num_items, num_genres), dtype=np.float32)
        for movie_id, g_vec in zip(movie_df['movie id'].values.tolist(), movie_df[genre_headers].values.tolist()):
            # check if movie_id was listed in ratings file and therefore in mapping dictionary
            if movie_id in v_dict.keys():
                v_features[v_dict[movie_id], :] = g_vec

        # user features

        sep = r'|'
        users_file = 'raw_data/' + dataset + '/u.user'
        users_headers = ['user id', 'age', 'gender', 'occupation', 'zip code']
        users_df = pd.read_csv(users_file, sep=sep, header=None,
                               names=users_headers, engine='python')

        occupation = set(users_df['occupation'].values.tolist())

        age = users_df['age'].values
        age_max = age.max()

        gender_dict = {'M': 0., 'F': 1.}
        occupation_dict = {f: i for i, f in enumerate(occupation, start=2)}

        num_feats = 2 + len(occupation_dict)

        u_features = np.zeros((num_users, num_feats), dtype=np.float32)
        for _, row in users_df.iterrows():
            u_id = row['user id']
            if u_id in u_dict.keys():
                # age
                u_features[u_dict[u_id], 0] = row['age'] / np.float(age_max)
                # gender
                u_features[u_dict[u_id], 1] = gender_dict[row['gender']]
                # occupation
                u_features[u_dict[u_id], occupation_dict[row['occupation']]] = 1.

    elif dataset == 'ml_1m':

        # load movie features
        movies_file = 'raw_data/' + dataset + '/movies.dat'

        movies_headers = ['movie_id', 'title', 'genre']
        movies_df = pd.read_csv(movies_file, sep=sep, header=None,
                                names=movies_headers, engine='python')

        # extracting all genres
        genres = []
        for s in movies_df['genre'].values:
            genres.extend(s.split('|'))

        genres = list(set(genres))
        num_genres = len(genres)

        genres_dict = {g: idx for idx, g in enumerate(genres)}

        # creating 0 or 1 valued features for all genres
        v_features = np.zeros((num_items, num_genres), dtype=np.float32)
        for movie_id, s in zip(movies_df['movie_id'].values.tolist(), movies_df['genre'].values.tolist()):
            # check if movie_id was listed in ratings file and therefore in mapping dictionary
            if movie_id in v_dict.keys():
                gen = s.split('|')
                for g in gen:
                    v_features[v_dict[movie_id], genres_dict[g]] = 1.

        # load user features
        users_file = 'raw_data/' + dataset + '/users.dat'
        users_headers = ['user_id', 'gender', 'age', 'occupation', 'zip-code']
        users_df = pd.read_csv(users_file, sep=sep, header=None,
                               names=users_headers, engine='python')

        # extracting all features
        cols = users_df.columns.values[1:]

        cntr = 0
        feat_dicts = []
        for header in cols:
            d = dict()
            feats = np.unique(users_df[header].values).tolist()
            d.update({f: i for i, f in enumerate(feats, start=cntr)})
            feat_dicts.append(d)
            cntr += len(d)

        num_feats = sum(len(d) for d in feat_dicts)

        u_features = np.zeros((num_users, num_feats), dtype=np.float32)
        for _, row in users_df.iterrows():
            u_id = row['user_id']
            if u_id in u_dict.keys():
                for k, header in enumerate(cols):
                    u_features[u_dict[u_id], feat_dicts[k][row[header]]] = 1.
    else:
        raise ValueError('Invalid dataset option %s' % dataset)

    u_features = sp.csr_matrix(u_features)
    v_features = sp.csr_matrix(v_features)

    print("User features shape: " + str(u_features.shape))
    print("Item features shape: " + str(v_features.shape))

    return u_features, v_features, rating_mx_train, train_labels, u_train_idx, v_train_idx, \
           val_labels, u_val_idx, v_val_idx, test_labels, u_test_idx, v_test_idx, class_values

