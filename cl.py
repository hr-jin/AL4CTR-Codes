import tensorflow as tf

def infonce_cosin_n(x_view_1, x_view_2, label_mask=None, tau=1.0, scope="up"):
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    x_view_1_norm = tf.math.l2_normalize(x_view_1, axis=-1)
    x_view_2_norm = tf.math.l2_normalize(x_view_2, axis=-1)

    pos_logits = tf.reduce_sum(tf.multiply(x_view_1_norm, x_view_2_norm), axis=-1)/tau
    
    pos_score = tf.exp(pos_logits)
    neg_logits = tf.matmul(x_view_1_norm, x_view_2_norm, transpose_b=True)/tau

    neg_score = tf.reduce_sum(tf.exp(neg_logits), axis=-1)

    if label_mask is not None:
      cl_loss = tf.losses.compute_weighted_loss(-tf.log(pos_score/neg_score), weights=label_mask)
    else:
      cl_loss = -tf.reduce_mean(tf.log(pos_score/neg_score))
  return cl_loss