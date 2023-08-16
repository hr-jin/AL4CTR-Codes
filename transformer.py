import numpy as np
import tensorflow as tf

def ln(inputs, epsilon=1e-8, scope="ln"):
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]

    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)

    beta = tf.get_variable(
      "beta",
      params_shape,
      initializer=tf.zeros_initializer()
    )

    gamma = tf.get_variable(
      "gamma",
      params_shape,
      initializer=tf.ones_initializer()
    )

    normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )

    outputs = gamma * normalized + beta

  return outputs

def mask_value(inputs, mask=None, mask_type=None):
  assert mask_type is not None
  padding_num = -2 ** 32 + 1
  if mask_type in ('k', 'key', 'keys'):
    mask = tf.to_float(mask)  # b*t
    mask = 1.0 - mask
    mask = tf.tile(mask, [tf.shape(inputs)[0] // tf.shape(mask)[0], 1]) # (b*n_heads)*t
    mask = tf.expand_dims(mask, axis=1)   # (b*n_heads)*1*t

    outputs = inputs + mask * padding_num

  elif mask_type in ('c', 'cau', 'causality'):
    diag_vals = tf.ones_like(inputs[0, :, :])
    tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
    causality_mask = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (b*n_heads)*t*t

    outputs = inputs + (1.0-causality_mask)*padding_num

  return outputs

def position_encoding(inputs, emb_dim, maxlen=75, scope="pos_enc"):
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    N = tf.shape(inputs)[0]
    T = tf.shape(inputs)[1]

    position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) # b*t
    position_enc = np.array([
      [pos / np.power(10000, (i-i%2)/emb_dim) for i in range(emb_dim)]
      for pos in range(maxlen)]
    )

    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    position_enc = tf.convert_to_tensor(position_enc, tf.float32) # (maxlen, E)

    outputs = tf.nn.embedding_lookup(position_enc, position_ind) # b*t*emb_dim

  return tf.to_float(outputs)

def position_encoding_learn(inputs, emb_dim, maxlen=75, scope="pos_enc_learn"):
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    N = tf.shape(inputs)[0]
    T = tf.shape(inputs)[1]
    pos_ind = tf.tile(tf.expand_dims(tf.cast(tf.range(T), tf.int64), 0), [N,1])

    pos_emb_table = tf.get_variable(
      "pos_embedding",
      shape=[maxlen, emb_dim],
      initializer=tf.initializers.truncated_normal(0, 0.05)
    )

    outputs = tf.nn.embedding_lookup(pos_emb_table, pos_ind)  # b*t*emb_dim

  return tf.to_float(outputs)

def session_encoding_learn(inputs, emb_dim, session_len=3, maxlen=25, scope="sess_enc_learn"):
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    N = tf.shape(inputs)[0]
    T = tf.shape(inputs)[1]
    pos_ind = tf.tile(tf.expand_dims(tf.cast(tf.range(T), tf.int64), 0), [N,1])

    pos_ind = pos_ind // session_len

    pos_emb_table = tf.get_variable(
      "pos_embedding",
      shape=[maxlen, emb_dim],
      initializer=tf.initializers.truncated_normal(0, 0.05)
    )

    outputs = tf.nn.embedding_lookup(pos_emb_table, pos_ind)  # b*t*emb_dim

  return tf.to_float(outputs)

def session_position_encoding_learn(inputs, emb_dim, session_len=3, scope="sess_pos_enc_learn"):
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    N = tf.shape(inputs)[0]
    T = tf.shape(inputs)[1]
    pos_ind = tf.tile(tf.expand_dims(tf.cast(tf.range(T), tf.int64), 0), [N,1])

    pos_ind = pos_ind % session_len

    pos_emb_table = tf.get_variable(
      "pos_embedding",
      shape=[session_len, emb_dim],
      initializer=tf.initializers.truncated_normal(0, 0.05)
    )

    outputs = tf.nn.embedding_lookup(pos_emb_table, pos_ind)  # b*t*emb_dim

  return tf.to_float(outputs)

def scaled_dot_product_attention(Q, K, V, key_masks, causality=False, dropout_rate=0.0, is_training=True, scope="sdpa"):
  d_k = Q.get_shape().as_list()[-1]
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    q_k = tf.matmul(Q, tf.transpose(K, perm=[0, 2, 1])) #(b*n_heads)*t*t
    q_k = q_k / (d_k ** 0.5)

    q_k = mask_value(q_k, key_masks, mask_type="key")

    if causality:
      q_k = mask_value(q_k, mask_type="causality")

    scores = tf.nn.softmax(q_k, axis=-1)

    scores = tf.layers.dropout(
      scores,
      rate=dropout_rate,
      training=is_training
    )

    outputs = tf.matmul(scores, V)

  return outputs

def multihead_attention(query, key, value, key_mask, num_heads=2, dropout_rate=0.0, is_training=True, causality=False, scope="mha"):
  d_model = query.get_shape().as_list()[-1]
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    Q = tf.layers.dense(  # b*t*d_model
      query,
      d_model,
      kernel_initializer=tf.initializers.truncated_normal(0, 0.05)
    )
    K = tf.layers.dense(  # b*t*d_model
      key,
      d_model,
      kernel_initializer=tf.initializers.truncated_normal(0, 0.05)
    )
    V = tf.layers.dense(  # b*t*d_model
      value,
      d_model,
      kernel_initializer=tf.initializers.truncated_normal(0, 0.05)
    )

    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  #(b*n_heads)*t*(d_model/n_heads)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  #(b*n_heads)*t*(d_model/n_heads)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  #(b*n_heads)*t*(d_model/n_heads)

    outputs = scaled_dot_product_attention(Q_, K_, V_, key_mask, causality, dropout_rate, is_training, scope="sdpa")

    outputs = tf.concat(tf.split(outputs, num_heads,axis=0), axis=2)

    outputs = tf.layers.dense(
      outputs,
      d_model,
      kernel_initializer=tf.initializers.truncated_normal(0, 0.05)
    )

  return outputs

def ffn(inputs, d_ff, scope="ffn"):
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    outputs = tf.layers.dense(
      inputs,
      d_ff[0],
      activation=tf.nn.relu,
      kernel_initializer=tf.initializers.truncated_normal(0, 0.05)
    )
    outputs = tf.layers.dense(
      outputs,
      d_ff[1],
      kernel_initializer=tf.initializers.truncated_normal(0, 0.05)
    )

  return outputs

def encoder(inputs, key_mask, num_heads, dropout_rate, d_ff, training, causality=False, scope="encoder"):
  d_model = inputs.get_shape().as_list()[-1]
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    attn_out = multihead_attention(
      inputs,
      inputs,
      inputs,
      key_mask,
      num_heads,
      dropout_rate,
      training,
      causality,
      scope="multi_head_attn"
    )
    attn_out = tf.layers.dropout(attn_out, dropout_rate, training=training)
    out1 = ln(inputs+attn_out, scope="ln_attn_out")

    ffn_out = ffn(out1, [d_ff, d_model], scope="ffn")
    ffn_out = tf.layers.dropout(ffn_out, dropout_rate, training=training)
    out2 = ln(out1 + ffn_out, scope="ln_ffn_out")

  return out2

def decoder(query, inputs, key_mask, num_heads, dropout_rate, d_ff, training, scope="decoder"):
  d_model = inputs.get_shape().as_list()[-1]
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    attn_out = multihead_attention(
      query,
      inputs,
      inputs,
      key_mask,
      num_heads,
      dropout_rate,
      training,
      False,
      scope="multi_head_attn"
    )
    attn_out = tf.layers.dropout(attn_out, dropout_rate, training=training)
    out1 = ln(query+attn_out, scope="ln_attn_out")

    ffn_out = ffn(out1, [d_ff, d_model], scope="ffn")
    ffn_out = tf.layers.dropout(ffn_out, dropout_rate, training=training)
    out2 = ln(out1 + ffn_out, scope="ln_ffn_out")

  return out2