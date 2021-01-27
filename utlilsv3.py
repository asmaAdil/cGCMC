from __future__ import division
from __future__ import print_function


def construct_feed_dict_e(placeholders, u_features, v_features, u_context, v_context, u_features_nonzero, v_features_nonzero,
                        support, support_t,support_e, support_e_t, labels, u_indices, v_indices, class_values,
                        dropout, u_features_side=None, v_features_side=None):

    print("Function that creates feed dictionary when running tensorflow sessions.")


    print(f" u_features {len(u_features)}")

    print(f"u_context {u_context.shape} v_context {v_context.shape}")
    print(f"  u_features_nonzero { u_features_nonzero} v_features_nonzero {v_features_nonzero} support {len(support)} ,  support_t {len(support_t)}")
    print(f"support_e {support_e.shape} support_e_t {support_e_t.shape} labels {labels.shape}")
    print(f" u_indices { u_indices.shape} v_indices {v_indices.shape} u_features_side {u_features_side} v_features_side {v_features_side} ")
    feed_dict = dict()

    feed_dict.update({placeholders['u_features']: u_features})
    feed_dict.update({placeholders['v_features']: v_features})


    feed_dict.update({placeholders['user_context']: u_context})
    feed_dict.update({placeholders['item_context']: v_context})


    feed_dict.update({placeholders['u_features_nonzero']: u_features_nonzero})
    feed_dict.update({placeholders['v_features_nonzero']: v_features_nonzero})

    # U x V
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['support_t']: support_t})

    # U x V x C
    feed_dict.update({placeholders['support_e']: support_e})
    feed_dict.update({placeholders['support_e_t']: support_e_t})

    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['user_indices']: u_indices})
    feed_dict.update({placeholders['item_indices']: v_indices})

    feed_dict.update({placeholders['dropout']: dropout})
    feed_dict.update({placeholders['class_values']: class_values})

    #no of features
    feed_dict.update({placeholders['u_features_side']: u_features_side})
    feed_dict.update({placeholders['v_features_side']: v_features_side})

    return feed_dict
