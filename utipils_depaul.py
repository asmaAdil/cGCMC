from __future__ import division
from __future__ import print_function


def construct_feed_dict_e(placeholders, e_f_u, e_f_v, u_features, v_features, u_context, v_context, u_features_nonzero, v_features_nonzero,
                        support, support_t, labels, u_indices, v_indices, class_values,
                        dropout):
    #
    """
    #,adj_context,adj_context_t
    Function that creates feed dictionary when running tensorflow sessions.


    print(f" u_features {len(u_features)}")

    print(f"u_context {u_context.shape} v_context {v_context.shape}")
    print(f"  u_features_nonzero { u_features_nonzero} v_features_nonzero {v_features_nonzero} support {len(support)} ,  support_t {len(support_t)}")
    print(f"labels {labels.shape}")
    print(f" u_indices {u_indices.shape} v_indices {v_indices.shape} u_features_side {u_features_side} v_features_side {v_features_side} ")
    """
    feed_dict = dict()

    print(f"problem feed Dic")
    feed_dict.update({placeholders['u_features']: u_features})
    feed_dict.update({placeholders['v_features']: v_features})


    feed_dict.update({placeholders['user_context']: u_context})
    feed_dict.update({placeholders['item_context']: v_context})

    feed_dict.update({placeholders['e_f_u']: e_f_u})
    feed_dict.update({placeholders['e_f_v']: e_f_v})

    print(f"problem feed Dic 2")
    feed_dict.update({placeholders['u_features_nonzero']: u_features_nonzero})
    feed_dict.update({placeholders['v_features_nonzero']: v_features_nonzero})

    # U x V
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['support_t']: support_t})

    print(f"problem feed Dic3")
    #feed_dict.update({placeholders['adj_context']: adj_context})
    #feed_dict.update({placeholders['adj_context_t']: adj_context_t})

    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['user_indices']: u_indices})
    feed_dict.update({placeholders['item_indices']: v_indices})

    feed_dict.update({placeholders['dropout']: dropout})
    feed_dict.update({placeholders['class_values']: class_values})



    print(f"problem feed Dic 4")
    return feed_dict
