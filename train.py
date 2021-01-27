""" Experiment runner for the model with knowledge graph attached to interaction data """

from __future__ import division
from __future__ import print_function

import argparse
import datetime
import time

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import scipy.sparse as sp
import sys
import json

from preprocessing import create_trainvaltest_split, \
    sparse_to_tuple, create_trainvaltest_split_Context, preprocess_user_item_features, globally_normalize_bipartite_adjacency, \
    load_data_monti, load_official_trainvaltest_split, normalize_features
from model import RecommenderGAE, RecommenderSideInfoGAE
from utils import construct_feed_dict


# Set random seed
# seed = 123 # use only for unit testing
seed = int(time.time())
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="ml_100k",
                choices=['ml_100k', 'ml_1m', 'ml_10m', 'douban', 'yahoo_music', 'flixster','LDOS'],
                help="Dataset string.")

ap.add_argument("-lr", "--learning_rate", type=float, default=0.01,
                help="Learning rate")

ap.add_argument("-e", "--epochs", type=int, default=2500,
                help="Number training epochs")

ap.add_argument("-hi", "--hidden", type=int, nargs=2, default=[500, 75],
                help="Number hidden units in 1st and 2nd layer")

ap.add_argument("-fhi", "--feat_hidden", type=int, default=64,
                help="Number hidden units in the dense layer for features")

ap.add_argument("-ehi", "--edge_hidden", type=int, default=10,
                help="Number hidden units in the dense layer for edge")

ap.add_argument("-ac", "--accumulation", type=str, default="sum", choices=['sum', 'stack'],
                help="Accumulation function: sum or stack.")

ap.add_argument("-do", "--dropout", type=float, default=0.7,
                help="Dropout fraction")

ap.add_argument("-nb", "--num_basis_functions", type=int, default=2,
                help="Number of basis functions for Mixture Model GCN.")

ap.add_argument("-ds", "--data_seed", type=int, default=1234,
                help="""Seed used to shuffle data in data_utils, taken from cf-nade (1234, 2341, 3412, 4123, 1324).
                     Only used for ml_1m and ml_10m datasets. """)

ap.add_argument("-sdir", "--summaries_dir", type=str, default='logs/' + str(datetime.datetime.now()).replace(' ', '_'),
                help="Directory for saving tensorflow summaries.")

# Boolean flags
fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('-nsym', '--norm_symmetric', dest='norm_symmetric',
                help="Option to turn on symmetric global normalization", action='store_true')
fp.add_argument('-nleft', '--norm_left', dest='norm_symmetric',
                help="Option to turn on left global normalization", action='store_false')
ap.set_defaults(norm_symmetric=True)

fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('-f', '--features', dest='features',
                help="Whether to use features (1) or not (0)", action='store_true')
fp.add_argument('-no_f', '--no_features', dest='features',
                help="Whether to use features (1) or not (0)", action='store_false')
ap.set_defaults(features=False)

fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('-ws', '--write_summary', dest='write_summary',
                help="Option to turn on summary writing", action='store_true')
fp.add_argument('-no_ws', '--no_write_summary', dest='write_summary',
                help="Option to turn off summary writing", action='store_false')
ap.set_defaults(write_summary=False)

fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('-t', '--testing', dest='testing',
                help="Option to turn on test set evaluation", action='store_true')
fp.add_argument('-v', '--validation', dest='testing',
                help="Option to only use validation set evaluation", action='store_false')
ap.set_defaults(testing=False)


args = vars(ap.parse_args())

print('Settings:')
print(args, '\n')

# Define parameters
DATASET = args['dataset']
DATASEED = args['data_seed']
NB_EPOCH = args['epochs']
DO = args['dropout']
HIDDEN = args['hidden']
FEATHIDDEN = args['feat_hidden']
EDGEHIDDEN = args['edge_hidden']
BASES = args['num_basis_functions']
LR = args['learning_rate']
WRITESUMMARY = args['write_summary']
SUMMARIESDIR = args['summaries_dir']
FEATURES = args['features']
SYM = args['norm_symmetric']
TESTING = args['testing']
ACCUM = args['accumulation']

SELFCONNECTIONS = False
SPLITFROMFILE = True
VERBOSE = True
FEATURES=True
DATASET == 'ml_100k'

if DATASET == 'ml_1m' or DATASET == 'ml_100k' or DATASET == 'douban'or DATASET == 'LDOS':
    NUMCLASSES = 5
elif DATASET == 'ml_10m':
    NUMCLASSES = 10
    print('\n WARNING: this might run out of RAM, consider using train_minibatch.py for dataset %s' % DATASET)
    print('If you want to proceed with this option anyway, uncomment this.\n')
    sys.exit(1)
elif DATASET == 'flixster':
    NUMCLASSES = 10
elif DATASET == 'yahoo_music':
    NUMCLASSES = 71
    if ACCUM == 'sum':
        print('\n WARNING: combining DATASET=%s with ACCUM=%s can cause memory issues due to large number of classes.')
        print('Consider using "--accum stack" as an option for this dataset.')
        print('If you want to proceed with this option anyway, uncomment this.\n')
        sys.exit(1)

# Splitting dataset in training, validation and test set

if DATASET == 'ml_1m' or DATASET == 'ml_10m':
    if FEATURES:
        datasplit_path = 'data/' + DATASET + '/withfeatures_split_seed' + str(DATASEED) + '.pickle'
    else:
        datasplit_path = 'data/' + DATASET + '/split_seed' + str(DATASEED) + '.pickle'
elif FEATURES:
    datasplit_path = 'data/' + DATASET + '/withfeatures.pickle'
    print(f"I am called and path is {datasplit_path}")
else:
    datasplit_path = 'data/' + DATASET + '/nofeatures.pickle'
print(f"DATASET {DATASET}")

if DATASET == 'flixster' or DATASET == 'douban' or DATASET == 'yahoo_music':
    u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
        val_labels, val_u_indices, val_v_indices, test_labels, \
        test_u_indices, test_v_indices, class_values = load_data_monti(DATASET, TESTING)

elif DATASET == 'ml_100k':
    print("Using official MovieLens dataset split u1.base/u1.test with 20% validation set size...")
    u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
        val_labels, val_u_indices, val_v_indices, test_labels, \
        test_u_indices, test_v_indices, class_values = load_official_trainvaltest_split(DATASET, TESTING)

elif DATASET=='LDOS':
    f=True #call load data in below function
    u_features, v_features, adj_train,e_features_train, train_labels, train_u_indices, train_v_indices, \
    val_labels,val_edge_f, val_u_indices, val_v_indices, test_labels, test_edge_f, \
    test_u_indices, test_v_indices, class_values = create_trainvaltest_split_Context(f,DATASET, DATASEED, TESTING,
                                                                             datasplit_path, SPLITFROMFILE,
                                                                             VERBOSE)

else:
    print("Using random dataset split ...")
    u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
        val_labels, val_u_indices, val_v_indices, test_labels, \
        test_u_indices, test_v_indices, class_values = create_trainvaltest_split(DATASET, DATASEED, TESTING,
                                                                                 datasplit_path, SPLITFROMFILE,
                                                                                 VERBOSE)

num_users, num_items = adj_train.shape

num_side_features = 0

FEATURES=True  #@asma
# feature loading
if not FEATURES:
    u_features = sp.identity(num_users, format='csr') #943 x 943
    v_features = sp.identity(num_items, format='csr')#(1682, 1682)
    u_features, v_features = preprocess_user_item_features(u_features, v_features)  #just stack
    #print(u_features.shape) #(943, 2625)
    #print(v_features.shape) #(1682, 2625)

elif FEATURES and u_features is not None and v_features is not None:
    # use features as side information and node_id's as node input features


    u_features_side = normalize_features(u_features)
    v_features_side = normalize_features(v_features)
    if e_features_train is not None:
        e_features_train_side=normalize_features(e_features_train)
        test_edge_f_side=normalize_features(test_edge_f)
        val_edge_f_side = normalize_features(val_edge_f)
        num_edge_side_features = e_features_train_side.shape[1]  # 49



    u_features_side, v_features_side = preprocess_user_item_features(u_features_side, v_features_side)  #943 x 23 and 1682 x 18 = (943, 41) and (1682, 41)

    u_features_side = np.array(u_features_side.todense(), dtype=np.float32) #(943, 41) (121, 2842)
    v_features_side = np.array(v_features_side.todense(), dtype=np.float32) # (1682, 41)  (1232, 2842)
    e_features_train_side = np.array(e_features_train_side.todense(), dtype=np.float32)
    test_edge_f_side = np.array(test_edge_f_side.todense(), dtype=np.float32)
    val_edge_f_side = np.array(val_edge_f_side.todense(), dtype=np.float32)


    num_side_features = u_features_side.shape[1]  #41 #2842

    # node id's for node input features
    id_csr_v = sp.identity(num_items, format='csr')
    id_csr_u = sp.identity(num_users, format='csr')

    u_features, v_features = preprocess_user_item_features(id_csr_u, id_csr_v) #943 x 943 (identity matrix) and v_features (1682 x 1682) (identity matrix) = (943, 2625) (1682, 2625) => stackede identity matrix


else:
    raise ValueError('Features flag is set to true but no features are loaded from dataset ' + DATASET)

# global normalization
support = []
support_t = []
adj_train_int = sp.csr_matrix(adj_train, dtype=np.int32) #(943, 1682) u v rating

for i in range(NUMCLASSES):
    # build individual binary rating matrices (supports) for each rating
    #print(f"i {i}")
    support_unnormalized = sp.csr_matrix(adj_train_int == i + 1, dtype=np.float32) #csr matrix 943 x 1682 only ontain no zero entries
    print(f"support_unnormalized {i} {support_unnormalized.shape}")
    #nnz Number of stored values, including explicit zeros.
    if support_unnormalized.nnz == 0 and DATASET != 'yahoo_music':
        # yahoo music has dataset split with not all ratings types present in training set.
        # this produces empty adjacency matrices for these ratings.
        sys.exit('ERROR: normalized bipartite adjacency matrix has only zero entries!!!!!')

    support_unnormalized_transpose = support_unnormalized.T
    support.append(support_unnormalized)
    support_t.append(support_unnormalized_transpose)

support = globally_normalize_bipartite_adjacency(support, symmetric=SYM)
support_t = globally_normalize_bipartite_adjacency(support_t, symmetric=SYM)

if SELFCONNECTIONS:  #set false by default
    support.append(sp.identity(u_features.shape[0], format='csr'))
    support_t.append(sp.identity(v_features.shape[0], format='csr'))
    print("self connection not called")

num_support = len(support)  #5
support = sp.hstack(support, format='csr') #64000
support_t = sp.hstack(support_t, format='csr')#64000
ACCUM='stack'

print(f"support {support.shape} {support_t.shape}")

if ACCUM == 'stack':
    div = HIDDEN[0] // num_support
    if HIDDEN[0] % num_support != 0:
        print("""\nWARNING: HIDDEN[0] (=%d) of stack layer is adjusted to %d such that
                  it can be evenly split in %d splits.\n""" % (HIDDEN[0], num_support * div, num_support))
    HIDDEN[0] = num_support * div


# Collect all user and item nodes for test set
test_u = list(set(test_u_indices))
test_v = list(set(test_v_indices))
test_u_dict = {n: i for i, n in enumerate(test_u)}
test_v_dict = {n: i for i, n in enumerate(test_v)}

test_u_indices = np.array([test_u_dict[o] for o in test_u_indices])
test_v_indices = np.array([test_v_dict[o] for o in test_v_indices])

test_support = support[np.array(test_u)]
test_support_t = support_t[np.array(test_v)]

#print(f"test_support {test_support.getnnz()}  test_support_t  {test_support_t.getnnz()}")
# Collect all user and item nodes for validation set
val_u = list(set(val_u_indices))
val_v = list(set(val_v_indices))
val_u_dict = {n: i for i, n in enumerate(val_u)}
val_v_dict = {n: i for i, n in enumerate(val_v)}

val_u_indices = np.array([val_u_dict[o] for o in val_u_indices])
val_v_indices = np.array([val_v_dict[o] for o in val_v_indices])

val_support = support[np.array(val_u)]
val_support_t = support_t[np.array(val_v)]

#print(f"val_support {val_support.getnnz()}  val_support_t  {val_support_t.getnnz()}")
# Collect all user and item nodes for train set
train_u = list(set(train_u_indices))
train_v = list(set(train_v_indices))
train_u_dict = {n: i for i, n in enumerate(train_u)}
train_v_dict = {n: i for i, n in enumerate(train_v)}

train_u_indices = np.array([train_u_dict[o] for o in train_u_indices])
train_v_indices = np.array([train_v_dict[o] for o in train_v_indices])

train_support = support[np.array(train_u)]
train_support_t = support_t[np.array(train_v)]

#print(f"train_support {train_support.getnnz()}  train_support_t   {train_support_t.getnnz()}")

# features as side info
if FEATURES:
    test_u_features_side = u_features_side[np.array(test_u)]
    test_v_features_side = v_features_side[np.array(test_v)]

    val_u_features_side = u_features_side[np.array(val_u)]
    val_v_features_side = v_features_side[np.array(val_v)]

    train_u_features_side = u_features_side[np.array(train_u)]
    train_v_features_side = v_features_side[np.array(train_v)]
    print("features set")

else:
    test_u_features_side = None
    test_v_features_side = None

    val_u_features_side = None
    val_v_features_side = None

    train_u_features_side = None
    train_v_features_side = None
    e_features_train_side = None

placeholders = {
    'u_features': tf.sparse_placeholder(tf.float32, shape=np.array(u_features.shape, dtype=np.int64)),
    'v_features': tf.sparse_placeholder(tf.float32, shape=np.array(v_features.shape, dtype=np.int64)),
    'u_features_nonzero': tf.placeholder(tf.int32, shape=()),
    'v_features_nonzero': tf.placeholder(tf.int32, shape=()),
    'labels': tf.placeholder(tf.int32, shape=(None,)),


    'u_features_side': tf.placeholder(tf.float32, shape=(None, num_side_features)),
    'v_features_side': tf.placeholder(tf.float32, shape=(None, num_side_features)),
    'e_features_side': tf.placeholder(tf.float32, shape=(None, num_edge_side_features)),


    'user_indices': tf.placeholder(tf.int32, shape=(None,)),
    'item_indices': tf.placeholder(tf.int32, shape=(None,)),

    'class_values': tf.placeholder(tf.float32, shape=class_values.shape),

    'dropout': tf.placeholder_with_default(0., shape=()),
    'weight_decay': tf.placeholder_with_default(0., shape=()),

    'support': tf.sparse_placeholder(tf.float32, shape=(None, None)),
    'support_t': tf.sparse_placeholder(tf.float32, shape=(None, None)),
}

do=0.7
nleft =True
nb=2
e=1000
features=True
FEATHIDDEN=10
EDGEHIDDEN=10
testing=True

if FEATURES:
    print("I am called RecommenderSideInfoGAE with features ")

    model = RecommenderSideInfoGAE(placeholders,
                                   input_dim=u_features.shape[1],
                                   feat_hidden_dim=FEATHIDDEN,
                                   edge_hidden_dim=EDGEHIDDEN,
                                   num_classes=NUMCLASSES,
                                   num_support=num_support,
                                   self_connections=SELFCONNECTIONS,
                                   num_basis_functions=BASES,
                                   hidden=HIDDEN,
                                   num_users=num_users,
                                   num_items=num_items,
                                   accum=ACCUM,
                                   learning_rate=LR,
                                   num_side_features=num_side_features,
                                   num_edge_side_features=num_edge_side_features,
                                   logging=True)
else:
    model = RecommenderGAE(placeholders,
                           input_dim=u_features.shape[1],
                           num_classes=NUMCLASSES,
                           num_support=num_support,
                           self_connections=SELFCONNECTIONS,
                           num_basis_functions=BASES,
                           hidden=HIDDEN,
                           num_users=num_users,
                           num_items=num_items,
                           accum=ACCUM,
                           learning_rate=LR,
                           logging=True)

print("Model build")

# Convert sparse placeholders to tuples to construct feed_dict
test_support = sparse_to_tuple(test_support)
test_support_t = sparse_to_tuple(test_support_t)

val_support = sparse_to_tuple(val_support)
val_support_t = sparse_to_tuple(val_support_t)

train_support = sparse_to_tuple(train_support)
train_support_t = sparse_to_tuple(train_support_t)

print(f"(u_features {u_features.shape} v_features {v_features.shape} ")

u_features = sparse_to_tuple(u_features)
v_features = sparse_to_tuple(v_features)



assert u_features[2][1] == v_features[2][1], 'Number of features of users and items must be the same!'

num_features = u_features[2][1]
#print(f"(num_features {u_features[0][2]} c  {u_features[1][1]} h  {u_features[2][1]})")

u_features_nonzero = u_features[1].shape[0]
v_features_nonzero = v_features[1].shape[0]



print("TRAINING TESTING VALIDATION CONSTRUCTING DICTIONARY")
# Feed_dicts for validation and test set stay constant over different update steps
train_feed_dict = construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero,
                                      v_features_nonzero, train_support, train_support_t,
                                      train_labels, train_u_indices, train_v_indices, class_values, DO,
                                      train_u_features_side, train_v_features_side, e_features_train_side)


# No dropout for validation and test runs
val_feed_dict = construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero,
                                    v_features_nonzero, val_support, val_support_t,
                                    val_labels, val_u_indices, val_v_indices, class_values, 0.,
                                    val_u_features_side, val_v_features_side,val_edge_f_side)



test_feed_dict = construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero,
                                     v_features_nonzero, test_support, test_support_t,
                                     test_labels, test_u_indices, test_v_indices, class_values, 0.,
                                     test_u_features_side, test_v_features_side, test_edge_f_side)


# Collect all variables to be logged into summary
merged_summary = tf.summary.merge_all()
"""
"""
sess = tf.Session()
sess.run(tf.global_variables_initializer()) #Returns an Op that initializes global variables.
#The FileWriter class provides a mechanism to create an event file in a given directory and add summaries and events to it.
# The class updates the file contents asynchronously.
# This allows a training program to call methods to add data to the file directly from the training loop, without slowing down training.

if WRITESUMMARY:
    train_summary_writer = tf.summary.FileWriter(SUMMARIESDIR + '/train', sess.graph)
    val_summary_writer = tf.summary.FileWriter(SUMMARIESDIR + '/val')
else:
    train_summary_writer = None
    val_summary_writer = None

best_val_score = np.inf
best_val_loss = np.inf
best_epoch = 0
wait = 0
TESTING=True

print('Training...')

NB_EPOCH=1
for epoch in range(NB_EPOCH):

    t = time.time()
    print(f"Time t{t}")

    # Run single weight update
    # outs = sess.run([model.opt_op, model.loss, model.rmse], feed_dict=train_feed_dict)
    # with exponential moving averages
    outs = sess.run([model.training_op, model.loss, model.rmse], feed_dict=train_feed_dict)
    print(f"all layers called and give output")

    train_avg_loss = outs[1]
    train_rmse = outs[2]


    val_avg_loss, val_rmse = sess.run([model.loss, model.rmse], feed_dict=val_feed_dict)


    if VERBOSE:
        print("[*] Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_avg_loss),
              "train_rmse=", "{:.5f}".format(train_rmse),
              "val_loss=", "{:.5f}".format(val_avg_loss),
              "val_rmse=", "{:.5f}".format(val_rmse),
              "\t\ttime=", "{:.5f}".format(time.time() - t))

    if val_rmse < best_val_score:
        best_val_score = val_rmse
        best_epoch = epoch
        print(f"best_val_score {best_val_score}   --- best_epoch {best_epoch}")


    if epoch % 20 == 0 and WRITESUMMARY:
        # Train set summary
        summary = sess.run(merged_summary, feed_dict=train_feed_dict)
        train_summary_writer.add_summary(summary, epoch)
        train_summary_writer.flush()

        # Validation set summary
        summary = sess.run(merged_summary, feed_dict=val_feed_dict)
        val_summary_writer.add_summary(summary, epoch)
        val_summary_writer.flush()

    if epoch % 100 == 0 and epoch > 1000 and not TESTING and False:
        saver = tf.train.Saver()
        save_path = saver.save(sess, "tmp/%s_seed%d.ckpt" % (model.name, DATASEED), global_step=model.global_step)

        # load polyak averages
        variables_to_restore = model.variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(sess, save_path)

        val_avg_loss, val_rmse = sess.run([model.loss, model.rmse], feed_dict=val_feed_dict)

        print('polyak val loss = ', val_avg_loss)
        print('polyak val rmse = ', val_rmse)

        # Load back normal variables
        saver = tf.train.Saver()
        saver.restore(sess, save_path)
        print("Testing is not false")


# store model including exponential moving averages
saver = tf.train.Saver()
save_path = saver.save(sess, "tmp/%s.ckpt" % model.name, global_step=model.global_step)
print(f"model.global_step {model.global_step}")

if VERBOSE:
    print("\nOptimization Finished!")
    print('best validation score =', best_val_score, 'at iteration', best_epoch)


if TESTING:
    print("Running Test seesion")
    test_avg_loss, test_rmse = sess.run([model.loss, model.rmse], feed_dict=test_feed_dict)
    print('test loss = ', test_avg_loss)
    print('test rmse = ', test_rmse)

    # restore with polyak averages of parameters
    # http://ruishu.io/2017/11/22/ema/
    variables_to_restore = model.variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, save_path)

    test_avg_loss, test_rmse = sess.run([model.loss, model.rmse], feed_dict=test_feed_dict)
    print('polyak test loss = ', test_avg_loss)
    print('polyak test rmse = ', test_rmse)

else:
    # restore with polyak averages of parameters
    variables_to_restore = model.variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, save_path)

    val_avg_loss, val_rmse = sess.run([model.loss, model.rmse], feed_dict=val_feed_dict)
    print('polyak val loss = ', val_avg_loss)
    print('polyak val rmse = ', val_rmse)

print('\nSETTINGS:\n')
for key, val in sorted(vars(ap.parse_args()).items()): #iteritems() python 2
    print(key, val)

print('global seed = ', seed)

# For parsing results from file
results = vars(ap.parse_args()).copy()
results.update({'best_val_score': float(best_val_score), 'best_epoch': best_epoch})
print(json.dumps(results))

sess.close()

