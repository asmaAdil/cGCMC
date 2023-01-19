from __future__ import print_function
from layers import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from metrics import softmax_accuracy, expected_rmse, softmax_cross_entropy, expected_mae

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.mae= 0
        self.optimizer = None
        self.opt_op = None
        self.global_step = tf.Variable(0, trainable=False)

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class RecommenderGAE(Model):
    def __init__(self, placeholders, input_dim, num_classes, num_support,
                 learning_rate, num_basis_functions, hidden, num_users, num_items, accum,
                 self_connections=False, **kwargs):
        super(RecommenderGAE, self).__init__(**kwargs)
        self.inputs = (placeholders['u_features'], placeholders['v_features'])
        self.u_features_nonzero = placeholders['u_features_nonzero']
        self.v_features_nonzero = placeholders['v_features_nonzero']
        self.support = placeholders['support']
        self.support_t = placeholders['support_t']
        self.dropout = placeholders['dropout']
        self.labels = placeholders['labels']
        self.u_indices = placeholders['user_indices']
        self.v_indices = placeholders['item_indices']
        self.class_values = placeholders['class_values']

        self.hidden = hidden
        self.num_basis_functions = num_basis_functions
        self.num_classes = num_classes
        self.num_support = num_support
        self.input_dim = input_dim
        self.self_connections = self_connections
        self.num_users = num_users
        self.num_items = num_items
        self.accum = accum
        self.learning_rate = learning_rate

        # standard settings: beta1=0.9, beta2=0.999, epsilon=1.e-8
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1.e-8)

        self.build()

        moving_average_decay = 0.995
        self.variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, self.global_step)
        self.variables_averages_op = self.variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([self.opt_op]):
            self.training_op = tf.group(self.variables_averages_op)

        self.embeddings = self.activations[2]

        print("**********************evaluations**********************")
        self._rmse()
        print("Ya Allah help in Getting MAE")
        self._mae()

    def _loss(self):
        self.loss += softmax_cross_entropy(self.outputs, self.labels)
        tf.summary.scalar('loss', self.loss)

    def _accuracy(self):
        self.accuracy = softmax_accuracy(self.outputs, self.labels)

    def _rmse(self):
        self.rmse = expected_rmse(self.outputs, self.labels, self.class_values)
        tf.summary.scalar('rmse_score', self.rmse)

    def _mae(self):
        self.mae = expected_mae(self.outputs, self.labels, self.class_values)
        tf.summary.scalar('mae_score', self.mae)

    def _build(self):
        if self.accum == 'sum':
            self.layers.append(OrdinalMixtureGCN(input_dim=self.input_dim,
                                                 output_dim=self.hidden[0],
                                                 support=self.support,
                                                 support_t=self.support_t,
                                                 num_support=self.num_support,
                                                 u_features_nonzero=self.u_features_nonzero,
                                                 v_features_nonzero=self.v_features_nonzero,
                                                 sparse_inputs=True,
                                                 act=tf.nn.relu,
                                                 bias=False,
                                                 dropout=self.dropout,
                                                 logging=self.logging,
                                                 share_user_item_weights=True,
                                                 self_connections=False))

        elif self.accum == 'stack':
            self.layers.append(StackGCN(input_dim=self.input_dim,
                                        output_dim=self.hidden[0],
                                        support=self.support,
                                        support_t=self.support_t,
                                        num_support=self.num_support,
                                        u_features_nonzero=self.u_features_nonzero,
                                        v_features_nonzero=self.v_features_nonzero,
                                        sparse_inputs=True,
                                        act=tf.nn.relu,
                                        dropout=self.dropout,
                                        logging=self.logging,
                                        share_user_item_weights=True))
        else:
            raise ValueError('accumulation function option invalid, can only be stack or sum.')

        self.layers.append(Dense(input_dim=self.hidden[0],
                                 output_dim=self.hidden[1],
                                 act=lambda x: x,
                                 dropout=self.dropout,
                                 logging=self.logging,
                                 share_user_item_weights=True))

        self.layers.append(BilinearMixture(num_classes=self.num_classes,
                                           u_indices=self.u_indices,
                                           v_indices=self.v_indices,
                                           input_dim=self.hidden[1],
                                           num_users=self.num_users,
                                           num_items=self.num_items,
                                           user_item_bias=False,
                                           dropout=0.,
                                           act=lambda x: x,
                                           num_weights=self.num_basis_functions,
                                           logging=self.logging,
                                           diagonal=False))


class RecommenderSideInfoGAE(Model):
    def __init__(self,  placeholders, input_dim, feat_hidden_dim, num_classes, num_support,
                 learning_rate, num_basis_functions, hidden, num_users, num_items, accum,
                 num_side_features, self_connections=False, **kwargs):
        super(RecommenderSideInfoGAE, self).__init__(**kwargs)

        self.inputs = (placeholders['u_features'], placeholders['v_features'])
        self.u_features_side = placeholders['u_features_side']
        self.v_features_side = placeholders['v_features_side']

        self.u_features_nonzero = placeholders['u_features_nonzero']
        self.v_features_nonzero = placeholders['v_features_nonzero']
        self.support = placeholders['support']
        self.support_t = placeholders['support_t']
        self.dropout = placeholders['dropout']
        self.labels = placeholders['labels']
        self.u_indices = placeholders['user_indices']
        self.v_indices = placeholders['item_indices']
        self.class_values = placeholders['class_values']

        self.num_side_features = num_side_features
        self.feat_hidden_dim = feat_hidden_dim
        if num_side_features > 0:
            self.u_features_side = placeholders['u_features_side']
            self.v_features_side = placeholders['v_features_side']

        else:
            self.u_features_side = None
            self.v_features_side = None

        self.hidden = hidden
        self.num_basis_functions = num_basis_functions
        self.num_classes = num_classes
        self.num_support = num_support
        self.input_dim = input_dim
        self.self_connections = self_connections
        self.num_users = num_users
        self.num_items = num_items
        self.accum = accum
        self.learning_rate = learning_rate

        # standard settings: beta1=0.9, beta2=0.999, epsilon=1.e-8
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1.e-8)

        self.build()

        moving_average_decay = 0.995
        self.variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, self.global_step)
        self.variables_averages_op = self.variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([self.opt_op]):
            self.training_op = tf.group(self.variables_averages_op)

        self.embeddings = self.activations[0]

        self._rmse()
        print("Ya Allah help in Getting MAE")
        self._mae()

    def _loss(self):
        self.loss += softmax_cross_entropy(self.outputs, self.labels)
        tf.summary.scalar('loss', self.loss)

    def _mae(self):
        self.mae = expected_mae(self.outputs, self.labels, self.class_values)
        tf.summary.scalar('mae_score', self.mae)

    def _accuracy(self):
        self.accuracy = softmax_accuracy(self.outputs, self.labels)

    def _rmse(self):
        self.rmse = expected_rmse(self.outputs, self.labels, self.class_values)

        tf.summary.scalar('rmse_score', self.rmse)

    def _build(self):
        if self.accum == 'sum':
            print(f"OrdinalMixtureGCN input dim : {self.input_dim}  output_dim {self.hidden[0]} ")
            self.layers.append(OrdinalMixtureGCN(input_dim=self.input_dim,
                                                 output_dim=self.hidden[0],
                                                 support=self.support,
                                                 support_t=self.support_t,
                                                 num_support=self.num_support,
                                                 u_features_nonzero=self.u_features_nonzero,
                                                 v_features_nonzero=self.v_features_nonzero,
                                                 sparse_inputs=True,
                                                 act=tf.nn.relu,
                                                 bias=False,
                                                 dropout=self.dropout,
                                                 logging=self.logging,
                                                 share_user_item_weights=True,
                                                 self_connections=self.self_connections))

        elif self.accum == 'stack':
            self.layers.append(StackGCN(input_dim=self.input_dim,
                                        output_dim=self.hidden[0],
                                        support=self.support,
                                        support_t=self.support_t,
                                        num_support=self.num_support,
                                        u_features_nonzero=self.u_features_nonzero,
                                        v_features_nonzero=self.v_features_nonzero,
                                        sparse_inputs=True,
                                        act=tf.nn.relu,
                                        dropout=self.dropout,
                                        logging=self.logging,
                                        share_user_item_weights=True))

        else:
            raise ValueError('accumulation function option invalid, can only be stack or sum.')

        print(f"Dense input dim : {self.num_side_features}  output_dim {self.feat_hidden_dim} ")
        self.layers.append(Dense(input_dim=self.num_side_features,
                                 output_dim=self.feat_hidden_dim,
                                 act=tf.nn.relu,
                                 dropout=0.,
                                 logging=self.logging,
                                 bias=True,
                                 share_user_item_weights=False))

        print(f"Dense input dim : {self.hidden[0]+self.feat_hidden_dim}  output_dim {self.hidden[1]} ")
        self.layers.append(Dense(input_dim=self.hidden[0]+self.feat_hidden_dim,
                                 output_dim=self.hidden[1],
                                 act=lambda x: x,
                                 dropout=self.dropout,
                                 logging=self.logging,
                                 share_user_item_weights=False))

        print(f"Dense input dim : {self.hidden[1]}  output_dim {self.num_classes} ")
        self.layers.append(BilinearMixture(num_classes=self.num_classes,
                                           u_indices=self.u_indices,
                                           v_indices=self.v_indices,
                                           input_dim=self.hidden[1],
                                           num_users=self.num_users,
                                           num_items=self.num_items,
                                           user_item_bias=False,
                                           dropout=0.,
                                           act=lambda x: x,
                                           num_weights=self.num_basis_functions,
                                           logging=self.logging,
                                           diagonal=False))

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build split sequential layer model

        # gcn layer
        layer = self.layers[0]
        gcn_hidden = layer(self.inputs)

        # dense layer for features
        layer = self.layers[1]
        feat_hidden = layer([self.u_features_side, self.v_features_side])

        # concat dense layer
        layer = self.layers[2]

        gcn_u = gcn_hidden[0]
        gcn_v = gcn_hidden[1]
        feat_u = feat_hidden[0]
        feat_v = feat_hidden[1]

        input_u = tf.concat(values=[gcn_u, feat_u], axis=1)
        input_v = tf.concat(values=[gcn_v, feat_v], axis=1)

        concat_hidden = layer([input_u, input_v])

        self.activations.append(concat_hidden)

        # Build sequential layer model
        for layer in self.layers[3::]:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step)


class RecommenderContextSideInfoGAE(Model):
    def __init__(self,  placeholders, input_dim, feat_hidden_dim, num_classes, num_support,
                 learning_rate, num_basis_functions, hidden, num_users, num_items, accum,
                 num_context_features, self_connections=False, **kwargs):
        super(RecommenderContextSideInfoGAE, self).__init__(**kwargs)

        self.inputs = (placeholders['u_features'], placeholders['v_features'])
        self.u_context_side = placeholders['user_context']
        self.v_context_side = placeholders['item_context']

        self.u_features_nonzero = placeholders['u_features_nonzero']
        self.v_features_nonzero = placeholders['v_features_nonzero']
        self.support = placeholders['support']
        self.support_t = placeholders['support_t']
        self.dropout = placeholders['dropout']
        self.labels = placeholders['labels']
        self.u_indices = placeholders['user_indices']
        self.v_indices = placeholders['item_indices']
        self.class_values = placeholders['class_values']
        self.e_f_u=placeholders['e_f_u']
        self.e_f_v=placeholders['e_f_v']

        self.num_context_features = num_context_features
        self.feat_hidden_dim = feat_hidden_dim
        if num_context_features > 0:
            self.u_context_side = placeholders['user_context']
            self.v_context_side = placeholders['item_context']

        else:
            self.u_context_side = None
            self.v_context_side = None

        self.hidden = hidden
        self.num_basis_functions = num_basis_functions
        self.num_classes = num_classes
        self.num_support = num_support
        self.input_dim = input_dim
        self.self_connections = self_connections
        self.num_users = num_users
        self.num_items = num_items
        self.accum = accum
        self.learning_rate = learning_rate

        # standard settings: beta1=0.9, beta2=0.999, epsilon=1.e-8
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1.e-8)

        self.build()

        moving_average_decay = 0.995
        self.variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, self.global_step)
        self.variables_averages_op = self.variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([self.opt_op]):
            self.training_op = tf.group(self.variables_averages_op)

        self.embeddings = self.activations[0]

        print("**********************evaluations**********************")
        self._rmse()

        self._mae()

    def _loss(self):
        self.loss += softmax_cross_entropy(self.outputs, self.labels)

        tf.summary.scalar('loss', self.loss)

    def _accuracy(self):
        self.accuracy = softmax_accuracy(self.outputs, self.labels)

    def _rmse(self):
        self.rmse = expected_rmse(self.outputs, self.labels, self.class_values)
        tf.summary.scalar('rmse_score', self.rmse)

    def _mae(self):
        self.mae = expected_mae(self.outputs, self.labels, self.class_values)
        tf.summary.scalar('mae_score', self.mae)

    def _build(self):
        if self.accum == 'sum':
            self.layers.append(OrdinalMixtureGCN3D(input_dim=self.input_dim,
                                                 output_dim=self.hidden[0],
                                                 support=self.support,
                                                 support_t=self.support_t,
                                                 num_support=self.num_support,
                                                 e_f_u=self.e_f_u,
                                                 e_f_v=self.e_f_v,
                                                 num_context=self.num_context_features,
                                                 u_features_nonzero=self.u_features_nonzero,
                                                 v_features_nonzero=self.v_features_nonzero,
                                                 sparse_inputs=True,
                                                 act=tf.nn.relu,
                                                 bias=False,
                                                 dropout=self.dropout,
                                                 logging=self.logging,
                                                 share_user_item_weights=True,
                                                 self_connections=self.self_connections))

        elif self.accum == 'stack':
            self.layers.append(StackGCN(input_dim=self.input_dim,
                                        output_dim=self.hidden[0],
                                        support=self.support,
                                        support_t=self.support_t,
                                        num_support=self.num_support,
                                        u_features_nonzero=self.u_features_nonzero,
                                        v_features_nonzero=self.v_features_nonzero,
                                        sparse_inputs=True,
                                        act=tf.nn.relu,
                                        dropout=self.dropout,
                                        logging=self.logging,
                                        share_user_item_weights=True))

        else:
            raise ValueError('accumulation function option invalid, can only be stack or sum.')

        self.layers.append(Dense(input_dim=self.num_context_features,
                                 output_dim=self.feat_hidden_dim,
                                 act=tf.nn.relu,
                                 dropout=0.,
                                 logging=self.logging,
                                 bias=True,
                                 share_user_item_weights=False))

        self.layers.append(Dense(input_dim=self.hidden[0]+self.feat_hidden_dim,
                                 output_dim=self.hidden[1],
                                 act=lambda x: x,
                                 dropout=self.dropout,
                                 logging=self.logging,
                                 share_user_item_weights=False))

        self.layers.append(BilinearMixture(num_classes=self.num_classes,
                                           u_indices=self.u_indices,
                                           v_indices=self.v_indices,
                                           input_dim=self.hidden[1],
                                           num_users=self.num_users,
                                           num_items=self.num_items,
                                           user_item_bias=False,
                                           dropout=0.,
                                           act=lambda x: x,
                                           num_weights=self.num_basis_functions,
                                           logging=self.logging,
                                           diagonal=False))

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build split sequential layer model
        print("Build-1")

        # gcn layer
        layer = self.layers[0]
        gcn_hidden = layer(self.inputs)


        # dense layer for features
        layer = self.layers[1]
        feat_hidden = layer([self.u_context_side, self.v_context_side])

        # concat dense layer
        layer = self.layers[2]

        gcn_u = gcn_hidden[0]
        gcn_v = gcn_hidden[1]


        feat_u = feat_hidden[0]
        feat_v = feat_hidden[1]
        print("Build-2")

        input_u = tf.concat(values=[gcn_u, feat_u], axis=1)
        input_v = tf.concat(values=[gcn_v, feat_v], axis=1)


        concat_hidden = layer([input_u, input_v])

        self.activations.append(concat_hidden)


        print("Build-4")
        # Build sequential layer model
        for layer in self.layers[3::]:

            hidden = layer(self.activations[-1])

            self.activations.append(hidden)

        self.outputs = self.activations[-1]


        self.outputs = self.activations[-1]


        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step)



class RecommenderBothSideInfoGAE(Model):
    def __init__(self,  placeholders, input_dim, feat_hidden_dim, num_classes, num_support,
                 learning_rate, num_basis_functions, hidden, num_users, num_items, accum,
                 num_side_features, self_connections=False,num_context_features=0,edge_hidden_dim=None, **kwargs):
        super(RecommenderBothSideInfoGAE, self).__init__(**kwargs)

        self.inputs = (placeholders['u_features'], placeholders['v_features'])
        self.u_features_side = placeholders['u_features_side']
        self.v_features_side = placeholders['v_features_side']

        self.u_features_nonzero = placeholders['u_features_nonzero']
        self.v_features_nonzero = placeholders['v_features_nonzero']
        self.support = placeholders['support']
        self.support_t = placeholders['support_t']
        self.dropout = placeholders['dropout']
        self.labels = placeholders['labels']
        self.u_indices = placeholders['user_indices']
        self.v_indices = placeholders['item_indices']
        self.class_values = placeholders['class_values']

        #self.e_f_u=placeholders['e_f_u']
        #self.e_f_v=placeholders['e_f_v']

        self.num_side_features = num_side_features
        self.feat_hidden_dim = feat_hidden_dim

        self.num_context_features = num_context_features
        self.edge_hidden_dim= edge_hidden_dim

        if num_side_features > 0:
            self.u_features_side = placeholders['u_features_side']
            self.v_features_side = placeholders['v_features_side']
        else:
            self.u_features_side = None
            self.v_features_side = None

        if num_context_features > 0:
            self.u_context_side = placeholders['user_context']
            self.v_context_side = placeholders['item_context']
        else:
            self.u_context_side = None
            self.v_context_side = None

        self.hidden = hidden
        self.num_basis_functions = num_basis_functions
        self.num_classes = num_classes
        self.num_support = num_support
        self.input_dim = input_dim
        self.self_connections = self_connections
        self.num_users = num_users
        self.num_items = num_items
        self.accum = accum
        self.learning_rate = learning_rate

        # standard settings: beta1=0.9, beta2=0.999, epsilon=1.e-8
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1.e-8)

        self.build()

        moving_average_decay = 0.995
        self.variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, self.global_step)
        self.variables_averages_op = self.variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([self.opt_op]):
            self.training_op = tf.group(self.variables_averages_op)

        self.embeddings = self.activations[0]

        print("**********************evaluations**********************")
        self._rmse()

        self._mae()


    def _loss(self):
        self.loss += softmax_cross_entropy(self.outputs, self.labels)
        tf.summary.scalar('loss', self.loss)

    def _accuracy(self):
        self.accuracy = softmax_accuracy(self.outputs, self.labels)

    def _rmse(self):
        self.rmse = expected_rmse(self.outputs, self.labels, self.class_values)
        tf.summary.scalar('rmse_score', self.rmse)

    def _mae(self):
        self.mae = expected_mae(self.outputs, self.labels, self.class_values)
        tf.summary.scalar('mae_score', self.mae)
    def _build(self):
        if self.accum == 'sum':
            print(f"OrdinalMixtureGCN input dim : {self.input_dim}  output_dim {self.hidden[0]} ")
            self.layers.append(OrdinalMixtureGCN(input_dim=self.input_dim,
                                                   output_dim=self.hidden[0],
                                                   support=self.support,
                                                   support_t=self.support_t,
                                                   num_support=self.num_support,
                                                   #e_f_u=self.e_f_u,
                                                   #e_f_v=self.e_f_v,
                                                   #num_context=self.num_context_features,
                                                   u_features_nonzero=self.u_features_nonzero,
                                                   v_features_nonzero=self.v_features_nonzero,
                                                   sparse_inputs=True,
                                                   act=tf.nn.relu,
                                                   bias=False,
                                                   dropout=self.dropout,
                                                   logging=self.logging,
                                                   share_user_item_weights=True,
                                                   self_connections=self.self_connections))
        elif self.accum == 'stack':
            self.layers.append(StackGCN(input_dim=self.input_dim,
                                        output_dim=self.hidden[0],
                                        support=self.support,
                                        support_t=self.support_t,
                                        num_support=self.num_support,
                                        e_f_u=self.e_f_u,
                                        e_f_v=self.e_f_v,
                                        num_context=self.num_context_features,
                                        u_features_nonzero=self.u_features_nonzero,
                                        v_features_nonzero=self.v_features_nonzero,
                                        sparse_inputs=True,
                                        act=tf.nn.relu,
                                        dropout=self.dropout,
                                        logging=self.logging,
                                        share_user_item_weights=True))

        else:
            raise ValueError('accumulation function option invalid, can only be stack or sum.')

        print(f"I am Dense feature input dim : {self.num_side_features}  output_dim {self.feat_hidden_dim} ")
        self.layers.append(Dense(input_dim=self.num_side_features,
                                 output_dim=self.feat_hidden_dim,
                                 act=tf.nn.relu,
                                 dropout=0.,
                                 logging=self.logging,
                                 bias=True,
                                 share_user_item_weights=False))

        print(f"I am Dense context input dim : {self.num_context_features}  output_dim {self.edge_hidden_dim} ")
        self.layers.append(Dense(input_dim=self.num_context_features,
                                 output_dim=self.edge_hidden_dim,
                                 act=tf.nn.relu,
                                 dropout=0.,
                                 logging=self.logging,
                                 bias=True,
                                 share_user_item_weights=False))

        print(f"Dense gcu +feat input dim : {self.hidden[0]+self.feat_hidden_dim}  output_dim {self.hidden[1]} ")
        self.layers.append(Dense(input_dim=self.hidden[0]+self.feat_hidden_dim,
                                 output_dim=self.hidden[2],
                                 act=lambda x: x,
                                 dropout=self.dropout,
                                 logging=self.logging,
                                 share_user_item_weights=False))

        print(f"Dense gcu + context input dim  : {self.hidden[0]+self.feat_hidden_dim}  output_dim {self.hidden[1]} ")
        self.layers.append(Dense(input_dim=self.hidden[0]+self.edge_hidden_dim,
                                 output_dim=self.hidden[2],
                                 act=lambda x: x,
                                 dropout=self.dropout,
                                 logging=self.logging,
                                 share_user_item_weights=False))

        print(f"Dense mixture context input dim : {self.hidden[1]+self.hidden[1]}  output_dim {self.hidden[1]} ")
        self.layers.append(Dense(input_dim=self.hidden[2]+self.hidden[2],
                                 output_dim=self.hidden[1],
                                 act=lambda x: x,
                                 dropout=self.dropout,
                                 logging=self.logging,
                                 share_user_item_weights=False))

        print(f"Dense input dim : {self.hidden[1]}  output_dim {self.num_classes} ")
        self.layers.append(BilinearMixture(num_classes=self.num_classes,
                                           u_indices=self.u_indices,
                                           v_indices=self.v_indices,
                                           input_dim=self.hidden[1],
                                           num_users=self.num_users,
                                           num_items=self.num_items,
                                           user_item_bias=False,
                                           dropout=0.,
                                           act=lambda x: x,
                                           num_weights=self.num_basis_functions,
                                           logging=self.logging,
                                           diagonal=False))

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build split sequential layer model

        # gcn layer

        print("convolution Layer 0")
        layer = self.layers[0]
        gcn_hidden = layer(self.inputs)

        # dense layer for features
        print("Feature Layer 1")
        layer = self.layers[1]
        feat_hidden = layer([self.u_features_side, self.v_features_side])

        # dense layer for context
        print("Context Layer 2")
        layer = self.layers[2]
        context_hidden = layer([self.u_context_side, self.v_context_side])

        # concat dense layer
        layer = self.layers[3]

        gcn_u = gcn_hidden[0]
        gcn_v = gcn_hidden[1]

        feat_u = feat_hidden[0]
        feat_v = feat_hidden[1]

        context_u=context_hidden[0]
        context_v = context_hidden[1]

        input_u = tf.concat(values=[gcn_u, feat_u], axis=1)
        input_v = tf.concat(values=[gcn_v, feat_v], axis=1)

        input_u_c = tf.concat(values=[gcn_u, context_u], axis=1)
        input_v_c = tf.concat(values=[gcn_v, context_v], axis=1)

        #input_u = tf.concat(values=[input_u, input_u_c], axis=1)
        #input_v = tf.concat(values=[input_v, input_v_c], axis=1)

        print(f"Input to Layer 3 context mix:")
        concat_hidden_c = layer([input_u_c, input_v_c])
        #self.activations.append(concat_hidden_c)

        print(f"Input to Layer 4 feature mix:")
        concat_hidden = layer([input_u, input_v])

        self.activations.append(concat_hidden)

        print(f"Input to Layer 5 all mix:")
        layer = self.layers[5]
        input_u_m = tf.concat(values=[concat_hidden_c[0], concat_hidden[0]], axis=1)
        input_v_m = tf.concat(values=[concat_hidden_c[1], concat_hidden[1]], axis=1)
        concat_mixture_c = layer([input_u_m, input_v_m])
        self.activations.append(concat_mixture_c)

        print(f"length self.activations {len(self.activations)}")

        # Build sequential layer model
        for layer in self.layers[6::]:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)

        print(f"length self.activationssss  {len(self.activations)}")
        self.outputs = self.activations[-1]

        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step)


class RecommenderContextSideInfoGAEv2(Model):
        def __init__(self, placeholders, input_dim, num_classes, num_support,
                     learning_rate, num_basis_functions, hidden, num_users, num_items, accum,
                     self_connections=False, **kwargs):
            super(RecommenderContextSideInfoGAEv2, self).__init__(**kwargs)

            self.inputs = (placeholders['u_features'], placeholders['v_features'])
            self.u_features_nonzero = placeholders['u_features_nonzero']
            self.v_features_nonzero = placeholders['v_features_nonzero']
            self.support = placeholders['support']
            self.support_t = placeholders['support_t']
            self.support_e = placeholders['support_e']
            self.support_e_t = placeholders['support_e_t']
            self.dropout = placeholders['dropout']
            self.labels = placeholders['labels']
            self.u_indices = placeholders['user_indices']
            self.v_indices = placeholders['item_indices']
            self.class_values = placeholders['class_values']

            self.hidden = hidden
            self.num_basis_functions = num_basis_functions
            self.num_classes = num_classes
            self.num_support = num_support
            self.input_dim = input_dim
            self.self_connections = self_connections
            self.num_users = num_users
            self.num_items = num_items
            self.accum = accum
            self.learning_rate = learning_rate

            # standard settings: beta1=0.9, beta2=0.999, epsilon=1.e-8
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                    epsilon=1.e-8)

            self.build()

            moving_average_decay = 0.995
            self.variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, self.global_step)
            self.variables_averages_op = self.variable_averages.apply(tf.trainable_variables())

            with tf.control_dependencies([self.opt_op]):
                self.training_op = tf.group(self.variables_averages_op)

            self.embeddings = self.activations[2]

            self._rmse()

        def _loss(self):
            self.loss += softmax_cross_entropy(self.outputs, self.labels)

            tf.summary.scalar('loss', self.loss)

        def _accuracy(self):
            self.accuracy = softmax_accuracy(self.outputs, self.labels)

        def _rmse(self):
            self.rmse = expected_rmse(self.outputs, self.labels, self.class_values)

            tf.summary.scalar('rmse_score', self.rmse)

        def _build(self):
            if self.accum == 'sum':
                self.layers.append(Ordinal3DMixtureGCN(input_dim=self.input_dim,
                                                     output_dim=self.hidden[0],
                                                     support=self.support,
                                                     support_t=self.support_t,
                                                     support_e=self.support_e,
                                                     support_e_t=self.support_e_t,
                                                     num_support=self.num_support,
                                                     u_features_nonzero=self.u_features_nonzero,
                                                     v_features_nonzero=self.v_features_nonzero,
                                                     sparse_inputs=True,
                                                     act=tf.nn.relu,
                                                     bias=False,
                                                     dropout=self.dropout,
                                                     logging=self.logging,
                                                     share_user_item_weights=True,
                                                     self_connections=False))

            elif self.accum == 'stack':
                self.layers.append(StackGCN(input_dim=self.input_dim,
                                            output_dim=self.hidden[0],
                                            support=self.support,
                                            support_t=self.support_t,
                                            support_e=self.support_e,
                                            support_e_t=self.support_e_t,
                                            num_support=self.num_support,
                                            u_features_nonzero=self.u_features_nonzero,
                                            v_features_nonzero=self.v_features_nonzero,
                                            sparse_inputs=True,
                                            act=tf.nn.relu,
                                            dropout=self.dropout,
                                            logging=self.logging,
                                            share_user_item_weights=True))
            else:
                raise ValueError('accumulation function option invalid, can only be stack or sum.')

            self.layers.append(Dense(input_dim=self.hidden[0],
                                     output_dim=self.hidden[1],
                                     act=lambda x: x,
                                     dropout=self.dropout,
                                     logging=self.logging,
                                     share_user_item_weights=True))

            self.layers.append(BilinearMixture(num_classes=self.num_classes,
                                               u_indices=self.u_indices,
                                               v_indices=self.v_indices,
                                               input_dim=self.hidden[1],
                                               num_users=self.num_users,
                                               num_items=self.num_items,
                                               user_item_bias=False,
                                               dropout=0.,
                                               act=lambda x: x,
                                               num_weights=self.num_basis_functions,
                                               logging=self.logging,
                                               diagonal=False))

class RecommenderContextConInfoGAE(Model):
    def __init__(self,  placeholders, input_dim, feat_hidden_dim, num_classes, num_support,
                 learning_rate, num_basis_functions, hidden, num_users, num_items, accum,
                 num_context_features, self_connections=False, **kwargs):
        super(RecommenderContextConInfoGAE, self).__init__(**kwargs)

        self.inputs = (placeholders['u_features'], placeholders['v_features'])
        self.u_context_side = placeholders['user_context']
        self.v_context_side = placeholders['item_context']

        self.u_features_nonzero = placeholders['u_features_nonzero']
        self.v_features_nonzero = placeholders['v_features_nonzero']
        self.support = placeholders['support']
        self.support_t = placeholders['support_t']
        self.dropout = placeholders['dropout']
        self.labels = placeholders['labels']
        self.u_indices = placeholders['user_indices']
        self.v_indices = placeholders['item_indices']
        self.class_values = placeholders['class_values']
        self.e_f_u=placeholders['e_f_u']
        self.e_f_v=placeholders['e_f_v']
        #self.adj_context=placeholders['adj_context']
        #self.adj_context_t=placeholders['adj_context_t']

        self.num_context_features = num_context_features
        self.feat_hidden_dim = feat_hidden_dim
        if num_context_features > 0:
            self.u_context_side = placeholders['user_context']
            self.v_context_side = placeholders['item_context']

        else:
            self.u_context_side = None
            self.v_context_side = None

        self.hidden = hidden
        self.num_basis_functions = num_basis_functions
        self.num_classes = num_classes
        self.num_support = num_support
        self.input_dim = input_dim
        self.self_connections = self_connections
        self.num_users = num_users
        self.num_items = num_items
        self.accum = accum
        self.learning_rate = learning_rate

        # standard settings: beta1=0.9, beta2=0.999, epsilon=1.e-8
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1.e-8)

        self.build()

        moving_average_decay = 0.995
        self.variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, self.global_step)
        self.variables_averages_op = self.variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([self.opt_op]):
            self.training_op = tf.group(self.variables_averages_op)

        self.embeddings = self.activations[0]

        self._rmse()
        print("Ya Allah help in Getting MAE")
        self._mae()

        print("*******************RecommenderContextConInfoGAE***********************")

    def _loss(self):
        self.loss += softmax_cross_entropy(self.outputs, self.labels)

        tf.summary.scalar('loss', self.loss)

    def _accuracy(self):
        self.accuracy = softmax_accuracy(self.outputs, self.labels)

    def _rmse(self):
        self.rmse = expected_rmse(self.outputs, self.labels, self.class_values)
        tf.summary.scalar('rmse_score', self.rmse)

    def _mae(self):
        self.mae = expected_mae(self.outputs, self.labels, self.class_values)
        tf.summary.scalar('mae_score', self.mae)

    def _build(self):
        if self.accum == 'sum':
            print(f"OrdinalMixtureGCN input dim {self.input_dim} output_dim {self.hidden[0]}")
            self.layers.append(OrdinalMixtureGCN(input_dim=self.input_dim,
                                                 output_dim=self.hidden[0],
                                                 support=self.support,
                                                 support_t=self.support_t,
                                                 num_support=self.num_support,
                                                 u_features_nonzero=self.u_features_nonzero,
                                                 v_features_nonzero=self.v_features_nonzero,
                                                 sparse_inputs=True,
                                                 act=tf.nn.relu,
                                                 bias=False,
                                                 dropout=self.dropout,
                                                 logging=self.logging,
                                                 share_user_item_weights=True,
                                                 self_connections=self.self_connections))



        elif self.accum == 'stack':
            self.layers.append(StackGCN(input_dim=self.input_dim,
                                        output_dim=self.hidden[0],
                                        support=self.support,
                                        support_t=self.support_t,
                                        num_support=self.num_support,
                                        u_features_nonzero=self.u_features_nonzero,
                                        v_features_nonzero=self.v_features_nonzero,
                                        sparse_inputs=True,
                                        act=tf.nn.relu,
                                        dropout=self.dropout,
                                        logging=self.logging,
                                        share_user_item_weights=True))

        else:
            raise ValueError('accumulation function option invalid, can only be stack or sum.')
        print(f"GCNC input dim {self.input_dim} output_dim {self.hidden[0]}")
        self.layers.append(GCNC(input_dim=self.input_dim,
                                output_dim=self.hidden[0],
                                adj_context=self.e_f_u,
                                adj_context_t=self.e_f_v,
                                u_features_nonzero=self.u_features_nonzero,
                                v_features_nonzero=self.v_features_nonzero,
                                sparse_inputs=True,
                                act=tf.nn.relu,
                                bias=False,
                                dropout=self.dropout,
                                logging=self.logging,
                                share_user_item_weights=True,
                                self_connections=self.self_connections))

        print(f"Dense input dim {self.num_context_features} output_dim {self.feat_hidden_dim}")
        self.layers.append(Dense(input_dim=self.num_context_features,
                                 output_dim=self.feat_hidden_dim,
                                 act=tf.nn.relu,
                                 dropout=0.,
                                 logging=self.logging,
                                 bias=True,
                                 share_user_item_weights=False))


        print(f"Dense input dim {self.hidden[0]+self.feat_hidden_dim } output_dim {self.hidden[1]}")
        self.layers.append(Dense(input_dim=self.hidden[0]+self.feat_hidden_dim,
                                 output_dim=self.hidden[2],
                                 act=lambda x: x,
                                 dropout=self.dropout,
                                 logging=self.logging,
                                 share_user_item_weights=False))
        print(f"Dense input dim {self.hidden[0] + self.hidden[0]} output_dim {self.hidden[2]}")
        self.layers.append(Dense(input_dim=self.hidden[0]+self.hidden[0],
                                 output_dim=self.hidden[2],
                                 act=lambda x: x,
                                 dropout=self.dropout,
                                 logging=self.logging,
                                 share_user_item_weights=False))
        print(f"Dense input dim {self.hidden[2] + self.hidden[2]} output_dim {self.hidden[1]}")
        self.layers.append(Dense(input_dim=self.hidden[2]+self.hidden[2],
                                 output_dim=self.hidden[1],
                                 act=lambda x: x,
                                 dropout=self.dropout,
                                 logging=self.logging,
                                 share_user_item_weights=False))
        print(f"Bilinear input dim {self.hidden[1]} output_dim {self.num_classes}")
        self.layers.append(BilinearMixture(num_classes=self.num_classes,
                                           u_indices=self.u_indices,
                                           v_indices=self.v_indices,
                                           input_dim=self.hidden[1],
                                           num_users=self.num_users,
                                           num_items=self.num_items,
                                           user_item_bias=False,
                                           dropout=0.,
                                           act=lambda x: x,
                                           num_weights=self.num_basis_functions,
                                           logging=self.logging,
                                           diagonal=False))

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build split sequential layer model
        print("Build-0: GCN")

        # gcn layer
        layer = self.layers[0]
        print(f"type self.inputs {type(self.inputs)}")
        gcn_hidden = layer(self.inputs)
        gcn_u = gcn_hidden[0]
        gcn_v = gcn_hidden[1]

        print("Build-1: GCNC")
        layer=self.layers[1]
        print(f"type self.inputs {type(self.inputs)}")
        gcnc_hidden=layer(self.inputs)
        gcnc_u = gcnc_hidden[0]
        gcnc_v = gcnc_hidden[1]

        print("Build-2: DENSE")
        # dense layer for features
        layer = self.layers[2]
        feat_hidden = layer([self.u_context_side, self.v_context_side])
        feat_u = feat_hidden[0]
        feat_v = feat_hidden[1]

        print("Build-3: DENSE")
        # concat dense layer
        layer = self.layers[3]
        input_u = tf.concat(values=[gcn_u, feat_u], axis=1)
        input_v = tf.concat(values=[gcn_v, feat_v], axis=1)
        concat_hidden = layer([input_u, input_v])
        self.activations.append(concat_hidden)

        print("Build-4")
        # concat dense layer
        layer = self.layers[4]
        input_uc = tf.concat(values=[gcn_u, gcnc_u], axis=1)
        input_vc = tf.concat(values=[gcn_v, gcnc_v], axis=1)
        concat_hidden_2 = layer([input_uc, input_vc])
        #self.activations.append(concat_hidden_2)

        print("Build-5")
        # concat dense layer
        layer = self.layers[5]
        print(f"concat_hidden[0] {concat_hidden[0].shape}, concat_hidden_2[0] {concat_hidden_2[0].shape}")
        input_u_m = tf.concat(values=[concat_hidden[0], concat_hidden_2[0]], axis=1)
        input_v_m = tf.concat(values=[concat_hidden[1], concat_hidden_2[1]], axis=1)
        print(f"concat_hidden[1] {concat_hidden[1].shape}, concat_hidden_2[1] {concat_hidden_2[1].shape}")
        concat_mixture_c = layer([input_u_m, input_v_m])
        self.activations.append(concat_mixture_c)

        print("Build-6")
        """
        input_u = tf.concat(values=[gcn_u, feat_u], axis=1)
        input_v = tf.concat(values=[gcn_v, feat_v], axis=1)

        input_u_c = tf.concat(values=[gcn_u, context_u], axis=1)
        input_v_c = tf.concat(values=[gcn_v, context_v], axis=1)

        #input_u = tf.concat(values=[input_u, input_u_c], axis=1)
        #input_v = tf.concat(values=[input_v, input_v_c], axis=1)

        print(f"Input to Layer 3 context mix:")
        concat_hidden_c = layer([input_u_c, input_v_c])
        #self.activations.append(concat_hidden_c)

        print(f"Input to Layer 4 feature mix:")
        concat_hidden = layer([input_u, input_v])

        self.activations.append(concat_hidden)

        print(f"Input to Layer 5 all mix:")
        layer = self.layers[5]
        input_u_m = tf.concat(values=[concat_hidden_c[0], concat_hidden[0]], axis=1)
        input_v_m = tf.concat(values=[concat_hidden_c[1], concat_hidden[1]], axis=1)
        concat_mixture_c = layer([input_u_m, input_v_m])
        self.activations.append(concat_mixture_c)

        print(f"length self.activations {len(self.activations)}")

        """


        # Build sequential layer model
        for layer in self.layers[6::]:

            hidden = layer(self.activations[-1])

            self.activations.append(hidden)

        self.outputs = self.activations[-1]


        self.outputs = self.activations[-1]


        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

class RecommenderBothSideConInfoGAE(Model):
    def __init__(self,  placeholders, input_dim, feat_hidden_dim, num_classes, num_support,
                 learning_rate, num_basis_functions, hidden, num_users, num_items, accum,
                 num_side_features, self_connections=False,num_context_features=0,edge_hidden_dim=None, **kwargs):
        super(RecommenderBothSideConInfoGAE, self).__init__(**kwargs)

        self.inputs = (placeholders['u_features'], placeholders['v_features'])
        self.u_features_side = placeholders['u_features_side']
        self.v_features_side = placeholders['v_features_side']

        self.u_features_nonzero = placeholders['u_features_nonzero']
        self.v_features_nonzero = placeholders['v_features_nonzero']
        self.support = placeholders['support']
        self.support_t = placeholders['support_t']
        self.dropout = placeholders['dropout']
        self.labels = placeholders['labels']
        self.u_indices = placeholders['user_indices']
        self.v_indices = placeholders['item_indices']
        self.class_values = placeholders['class_values']
        self.e_f_u=placeholders['e_f_u']
        self.e_f_v=placeholders['e_f_v']

        self.num_side_features = num_side_features
        self.feat_hidden_dim = feat_hidden_dim

        self.num_context_features = num_context_features
        self.edge_hidden_dim= edge_hidden_dim

        if num_side_features > 0:
            self.u_features_side = placeholders['u_features_side']
            self.v_features_side = placeholders['v_features_side']
        else:
            self.u_features_side = None
            self.v_features_side = None

        if num_context_features > 0:
            self.u_context_side = placeholders['user_context']
            self.v_context_side = placeholders['item_context']
        else:
            self.u_context_side = None
            self.v_context_side = None

        self.hidden = hidden
        self.num_basis_functions = num_basis_functions
        self.num_classes = num_classes
        self.num_support = num_support
        self.input_dim = input_dim
        self.self_connections = self_connections
        self.num_users = num_users
        self.num_items = num_items
        self.accum = accum
        self.learning_rate = learning_rate


        # standard settings: beta1=0.9, beta2=0.999, epsilon=1.e-8
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1.e-8)

        self.build()

        moving_average_decay = 0.995
        self.variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, self.global_step)
        self.variables_averages_op = self.variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([self.opt_op]):
            self.training_op = tf.group(self.variables_averages_op)

        self.embeddings = self.activations[0]

        self._rmse()
        print("Ya Allah help in Getting MAE")
        self._mae()

    def _loss(self):
        self.loss += softmax_cross_entropy(self.outputs, self.labels)
        tf.summary.scalar('loss', self.loss)

    def _mae(self):
        self.mae = expected_mae(self.outputs, self.labels, self.class_values)
        tf.summary.scalar('mae_score', self.mae)

    def _accuracy(self):
        self.accuracy = softmax_accuracy(self.outputs, self.labels)

    def _rmse(self):
        self.rmse = expected_rmse(self.outputs, self.labels, self.class_values)

        tf.summary.scalar('rmse_score', self.rmse)

    def _build(self):
        if self.accum == 'sum':
            print(f"OrdinalMixtureGCN input dim : {self.input_dim}  output_dim {self.hidden[0]} ")
            self.layers.append(OrdinalMixtureGCN(input_dim=self.input_dim,
                                                   output_dim=self.hidden[0],
                                                   support=self.support,
                                                   support_t=self.support_t,
                                                   num_support=self.num_support,
                                                   #e_f_u=self.e_f_u,
                                                   #e_f_v=self.e_f_v,
                                                   #num_context=self.num_context_features,
                                                   u_features_nonzero=self.u_features_nonzero,
                                                   v_features_nonzero=self.v_features_nonzero,
                                                   sparse_inputs=True,
                                                   act=tf.nn.relu,
                                                   bias=False,
                                                   dropout=self.dropout,
                                                   logging=self.logging,
                                                   share_user_item_weights=True,
                                                   self_connections=self.self_connections))
        elif self.accum == 'stack':
            self.layers.append(StackGCN(input_dim=self.input_dim,
                                        output_dim=self.hidden[0],
                                        support=self.support,
                                        support_t=self.support_t,
                                        num_support=self.num_support,
                                        e_f_u=self.e_f_u,
                                        e_f_v=self.e_f_v,
                                        num_context=self.num_context_features,
                                        u_features_nonzero=self.u_features_nonzero,
                                        v_features_nonzero=self.v_features_nonzero,
                                        sparse_inputs=True,
                                        act=tf.nn.relu,
                                        dropout=self.dropout,
                                        logging=self.logging,
                                        share_user_item_weights=True))

        else:
            raise ValueError('accumulation function option invalid, can only be stack or sum.')

        print(f"GCNC input dim {self.input_dim} output_dim {self.hidden[0]}")
        self.layers.append(GCNC(input_dim=self.input_dim,
                                output_dim=self.hidden[0],
                                adj_context=self.e_f_u,
                                adj_context_t=self.e_f_v,
                                u_features_nonzero=self.u_features_nonzero,
                                v_features_nonzero=self.v_features_nonzero,
                                sparse_inputs=True,
                                act=tf.nn.relu,
                                bias=False,
                                dropout=self.dropout,
                                logging=self.logging,
                                share_user_item_weights=True,
                                self_connections=self.self_connections))

        print(f"I am Dense feature input dim : {self.num_side_features}  output_dim {self.feat_hidden_dim} ")
        self.layers.append(Dense(input_dim=self.num_side_features,
                                 output_dim=self.feat_hidden_dim,
                                 act=tf.nn.relu,
                                 dropout=0.,
                                 logging=self.logging,
                                 bias=True,
                                 share_user_item_weights=False))

        print(f"I am Dense context input dim : {self.num_context_features}  output_dim {self.edge_hidden_dim} ")
        self.layers.append(Dense(input_dim=self.num_context_features,
                                 output_dim=self.edge_hidden_dim,
                                 act=tf.nn.relu,
                                 dropout=0.,
                                 logging=self.logging,
                                 bias=True,
                                 share_user_item_weights=False))
        print(f"Dense gcu + gcc  : {self.hidden[0] + self.hidden[0]}  output_dim {self.hidden[1]} ")
        self.layers.append(Dense(input_dim=self.hidden[0] + self.hidden[0] +self.feat_hidden_dim + self.edge_hidden_dim,
                                 output_dim=self.hidden[2],
                                 act=lambda x: x,
                                 dropout=self.dropout,
                                 logging=self.logging,
                                 share_user_item_weights=False))

        print(f"Dense gcu + gcc  : {self.hidden[2]}  output_dim {self.hidden[1]} ")
        self.layers.append(Dense(input_dim=self.hidden[2] ,
                                 output_dim=self.hidden[1],
                                 act=lambda x: x,
                                 dropout=self.dropout,
                                 logging=self.logging,
                                 share_user_item_weights=False))
        """

        print(f"Dense gcu + gcc  : {self.hidden[0]+ self.hidden[0] }  output_dim {self.hidden[1]} ")
        self.layers.append(Dense(input_dim=self.hidden[0]+self.hidden[0],
                                 output_dim=self.hidden[2],
                                 act=lambda x: x,
                                 dropout=self.dropout,
                                 logging=self.logging,
                                 share_user_item_weights=False))

        print(f"Dense gcu +feat input dim : {self.hidden[0]+self.feat_hidden_dim}  output_dim {self.hidden[1]} ")
        self.layers.append(Dense(input_dim=self.hidden[0]+self.feat_hidden_dim,
                                 output_dim=self.hidden[2],
                                 act=lambda x: x,
                                 dropout=self.dropout,
                                 logging=self.logging,
                                 share_user_item_weights=False))

        print(f"Dense gcu + context input dim  : {self.hidden[0]+self.feat_hidden_dim}  output_dim {self.hidden[1]} ")
        self.layers.append(Dense(input_dim=self.hidden[0]+self.edge_hidden_dim,
                                 output_dim=self.hidden[2],
                                 act=lambda x: x,
                                 dropout=self.dropout,
                                 logging=self.logging,
                                 share_user_item_weights=False))


        print(f"Dense mixture context input dim : {self.hidden[1]+self.hidden[1]}  output_dim {self.hidden[1]} ")
        self.layers.append(Dense(input_dim=self.hidden[2]+self.hidden[2],
                                 output_dim=self.hidden[1],
                                 act=lambda x: x,
                                 dropout=self.dropout,
                                 logging=self.logging,
                                 share_user_item_weights=False))

        print(f"Dense mixture context input dim : {self.hidden[1]+self.hidden[1]}  output_dim {self.hidden[1]} ")
        self.layers.append(Dense(input_dim=self.hidden[2]+self.hidden[2],
                                 output_dim=self.hidden[1],
                                 act=lambda x: x,
                                 dropout=self.dropout,
                                 logging=self.logging,
                                 share_user_item_weights=False))


        print(f"Dense mixture context input dim : {self.hidden[1]+self.hidden[1]}  output_dim {self.hidden[1]} ")
        self.layers.append(Dense(input_dim=self.hidden[1]+self.hidden[1],
                                 output_dim=self.hidden[1],
                                 act=lambda x: x,
                                 dropout=self.dropout,
                                 logging=self.logging,
                                 share_user_item_weights=False))
        """
        print(f"Dense input dim : {self.hidden[1]}  output_dim {self.num_classes} ")
        self.layers.append(BilinearMixture(num_classes=self.num_classes,
                                           u_indices=self.u_indices,
                                           v_indices=self.v_indices,
                                           input_dim=self.hidden[1],
                                           num_users=self.num_users,
                                           num_items=self.num_items,
                                           user_item_bias=False,
                                           dropout=0.,
                                           act=lambda x: x,
                                           num_weights=self.num_basis_functions,
                                           logging=self.logging,
                                           diagonal=False))

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build split sequential layer model

        # gcn layer

        print("Build 0  RATING")
        layer = self.layers[0]
        gcn_hidden = layer(self.inputs)
        gcn_u = gcn_hidden[0]
        gcn_v = gcn_hidden[1]

        print("Build 1  CONTEXT")
        layer = self.layers[1]
        gcn_c_hidden = layer(self.inputs)
        gcn_cu = gcn_c_hidden[0]
        gcn_cv = gcn_c_hidden[1]

        # dense layer for features
        print("Build-2  FEATURES")
        layer = self.layers[2]
        feat_hidden = layer([self.u_features_side, self.v_features_side])
        feat_u = feat_hidden[0]
        feat_v = feat_hidden[1]

        # dense layer for context
        print("build-3")
        layer = self.layers[3]
        context_hidden = layer([self.u_context_side, self.v_context_side])
        context_u=context_hidden[0]
        context_v = context_hidden[1]

        print("Build-4: DENSE")
        # concat dense layer
        layer = self.layers[4]
        input_u_f = tf.concat(values=[gcn_u, gcn_cu, feat_u, context_u], axis=1)
        input_v_f = tf.concat(values=[gcn_v, gcn_cv, feat_v, context_v], axis=1)
        concat_hidden_f = layer([input_u_f, input_v_f])
        self.activations.append(concat_hidden_f)

        """
        print("Build-4: DENSE")
        # concat dense layer
        layer = self.layers[4]
        input_uc = tf.concat(values=[gcn_u, gcn_cu], axis=1)
        input_vc = tf.concat(values=[gcn_v, gcn_cv], axis=1)
        concat_hidden_1 = layer([input_uc, input_vc])
        self.activations.append(concat_hidden_1)

        print("Build-5: DENSE")
        # concat dense layer
        layer = self.layers[5]
        input_uf = tf.concat(values=[gcn_u, feat_u], axis=1)
        input_vf = tf.concat(values=[gcn_v, feat_v], axis=1)
        concat_hidden_2 = layer([input_uf, input_vf])
        self.activations.append(concat_hidden_2)

        print("Build-6: DENSE")
        # concat dense layer
        layer = self.layers[6]
        input_u_cu = tf.concat(values=[gcn_u, context_u], axis=1)
        input_v_cv = tf.concat(values=[gcn_v, context_v], axis=1)
        concat_hidden_3 = layer([input_u_cu, input_v_cv])

        print("Build-7: DENSE")
        # concat dense layer
        layer = self.layers[7]
        input_u_ff = tf.concat(values=[concat_hidden_1[0], concat_hidden_2[0]], axis=1)
        input_v_ff = tf.concat(values=[concat_hidden_1[1], concat_hidden_2[1]], axis=1)
        concat_hidden_4 = layer([input_u_ff, input_v_ff])

        print("Build-8: DENSE")
        # concat dense layer
        layer = self.layers[8]
        input_u_ff = tf.concat(values=[concat_hidden_1[0], concat_hidden_3[0]], axis=1)
        input_v_ff = tf.concat(values=[concat_hidden_1[1], concat_hidden_3[1]], axis=1)
        concat_hidden_5 = layer([input_u_ff, input_v_ff])

        print("Build-9: DENSE")
        # concat dense layer
        layer = self.layers[9]
        input_u_f = tf.concat(values=[concat_hidden_4[0], concat_hidden_5[0]], axis=1)
        input_v_f = tf.concat(values=[concat_hidden_4[1], concat_hidden_5[1]], axis=1)
        concat_hidden_f= layer([input_u_f, input_v_f])

        self.activations.append(concat_hidden_f)
        """
        print(f"length self.activations {len(self.activations)}")

        # Build sequential layer model
        for layer in self.layers[5::]:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)

        print(f"length self.activationssss  {len(self.activations)}")
        self.outputs = self.activations[-1]

        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
