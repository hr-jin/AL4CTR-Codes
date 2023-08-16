# coding=utf-8
import math
import numpy as np

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell

import cl
from utils import *
from Dice import dice
from rnn import dynamic_rnn
import transformer

#### CAN config #####
weight_emb_w = [[16, 8], [8, 4]]
weight_emb_b = [0, 0]
print(weight_emb_w, weight_emb_b)
orders = 3
order_indep = False  # True
WEIGHT_EMB_DIM = (sum([w[0] * w[1] for w in weight_emb_w]) + sum(weight_emb_b))  # * orders
INDEP_NUM = 1
if order_indep:
  INDEP_NUM *= orders

print("orders: ", orders)
CALC_MODE = "can"
device = '/gpu:0'


#### CAN config #####

def gen_coaction(ad, his_items, dim, mode="can", mask=None):
  weight, bias = [], []
  idx = 0
  weight_orders = []
  bias_orders = []
  for i in range(orders):
    for w, b in zip(weight_emb_w, weight_emb_b):
      weight.append(tf.reshape(ad[:, idx:idx + w[0] * w[1]], [-1, w[0], w[1]]))
      idx += w[0] * w[1]
      if b == 0:
        bias.append(None)
      else:
        bias.append(tf.reshape(ad[:, idx:idx + b], [-1, 1, b]))
        idx += b
    weight_orders.append(weight)
    bias_orders.append(bias)
    if not order_indep:
      break

  if mode == "can":
    out_seq = []
    hh = []
    for i in range(orders):
      hh.append(his_items ** (i + 1))
    # hh = [sum(hh)]
    for i, h in enumerate(hh):
      if order_indep:
        weight, bias = weight_orders[i], bias_orders[i]
      else:
        weight, bias = weight_orders[0], bias_orders[0]
      for j, (w, b) in enumerate(zip(weight, bias)):
        h = tf.matmul(h, w)
        if b is not None:
          h = h + b
        if j != len(weight) - 1:
          h = tf.nn.tanh(h)
        out_seq.append(h)
    out_seq = tf.concat(out_seq, 2)
    if mask is not None:
      mask = tf.expand_dims(mask, axis=-1)
      out_seq = out_seq * mask
  out = tf.reduce_sum(out_seq, 1)
  # if keep_fake_carte_seq and mode=="emb":
  #    return out, out_seq
  return out, None


class Model(object):
  def __init__(self, n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False,
               use_softmax=True, use_coaction=False, use_cartes=False, use_infonce=False, ui_infonce_tau=0.07,
               aux_infonce_w=0.05, ui_proj_dim=16, 
               use_lal_infonce=False, lal_infonce_tau=0.07, lal_infonce_w=0.01,
               infonce_loss_type="contrastive",
               model_type="DBPMaN"):
    with tf.name_scope('Inputs'):
      self.mid_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='mid_his_batch_ph')
      self.cate_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='cate_his_batch_ph')
      self.uid_batch_ph = tf.placeholder(tf.int32, [None, ], name='uid_batch_ph')
      self.mid_batch_ph = tf.placeholder(tf.int32, [None, ], name='mid_batch_ph')
      self.cate_batch_ph = tf.placeholder(tf.int32, [None, ], name='cate_batch_ph')
      self.mask = tf.placeholder(tf.float32, [None, None], name='mask')

      self.mid_sess_his = tf.placeholder(tf.int32, [None, 18, 10], name='mid_sess_his')  # [1024, 18, 10]
      self.cat_sess_his = tf.placeholder(tf.int32, [None, 18, 10], name='cat_sess_his')
      self.sess_mask = tf.placeholder(tf.int32, [None, 18], name='sess_mask')
      self.mid_sess_tgt = tf.placeholder(tf.int32, [None, 18], name='mid_sess_tgt')  # [1024, 18]
      self.cat_sess_tgt = tf.placeholder(tf.int32, [None, 18], name='cat_sess_tgt')
      self.fin_mid_sess = tf.placeholder(tf.int32, [None, 10], name='fin_mid_sess')  # [1024, 10]
      self.fin_cat_sess = tf.placeholder(tf.int32, [None, 10], name='fin_cat_sess')

      self.seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')
      self.target_ph = tf.placeholder(tf.float32, [None, None], name='target_ph')

      self.lr = tf.placeholder(tf.float64, [])

      self.use_negsampling = use_negsampling
      self.use_softmax = False  # use_softmax
      self.use_coaction = use_coaction
      self.use_cartes = use_cartes

      self.use_infonce = use_infonce
      self.ui_infonce_tau = ui_infonce_tau
      self.aux_infonce_w = aux_infonce_w
      self.ui_proj_dim = ui_proj_dim

      self.model_type = model_type

      self.use_lal_infonce = use_lal_infonce
      self.lal_infonce_tau = lal_infonce_tau
      self.lal_infonce_w = lal_infonce_w

      self.infonce_loss_type = infonce_loss_type

      print("args:")
      print("negsampling: ", self.use_negsampling)
      print("softmax: ", self.use_softmax)
      print("co-action: ", self.use_coaction)
      print("carte: ", self.use_cartes)
      print("infonce: ", self.use_infonce)
      if use_negsampling:
        self.noclk_mid_batch_ph = tf.placeholder(tf.int32, [None, None, None],
                                                 name='noclk_mid_batch_ph')  # generate 3 item IDs from negative sampling.
        self.noclk_cate_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='noclk_cate_batch_ph')

    # Embedding layer
    with tf.name_scope('Embedding_layer'):
      self.uid_embeddings_var = tf.get_variable("uid_embedding_var", [n_uid, EMBEDDING_DIM])
      self.uid_batch_embedded = tf.nn.embedding_lookup(self.uid_embeddings_var, self.uid_batch_ph)
      if "FM" in self.model_type:
        self.uid_embeddings_var_1dim = tf.get_variable("uid_embedding_var_1dim", [n_uid, 1])
        self.uid_batch_embedded_1dim = tf.nn.embedding_lookup(self.uid_embeddings_var_1dim, self.uid_batch_ph)

      self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [n_mid, EMBEDDING_DIM])
      self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)
      self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph)
      if "FM" in self.model_type:
        self.mid_embeddings_var_1dim = tf.get_variable("mid_embedding_var_1dim", [n_mid, 1])
        self.mid_batch_embedded_1dim = tf.nn.embedding_lookup(self.mid_embeddings_var_1dim, self.mid_batch_ph)
        self.mid_his_batch_embedded_1dim = tf.nn.embedding_lookup(self.mid_embeddings_var_1dim, self.mid_his_batch_ph)

      if self.use_negsampling:
        self.noclk_mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var,
                                                                   self.noclk_mid_batch_ph)

      self.cate_embeddings_var = tf.get_variable("cate_embedding_var", [n_cate, EMBEDDING_DIM])
      self.cate_batch_embedded = tf.nn.embedding_lookup(self.cate_embeddings_var, self.cate_batch_ph)
      self.cate_his_batch_embedded = tf.nn.embedding_lookup(self.cate_embeddings_var, self.cate_his_batch_ph)
      if "FM" in self.model_type:
        self.cate_embeddings_var_1dim = tf.get_variable("cate_embedding_var_1dim", [n_cate, 1])
        self.cate_batch_embedded_1dim = tf.nn.embedding_lookup(self.cate_embeddings_var_1dim, self.cate_batch_ph)
        self.cate_his_batch_embedded_1dim = tf.nn.embedding_lookup(self.cate_embeddings_var_1dim, self.cate_his_batch_ph)

      if self.use_negsampling:
        self.noclk_cate_his_batch_embedded = tf.nn.embedding_lookup(self.cate_embeddings_var,
                                                                    self.noclk_cate_batch_ph)

      ###  co-action ###
      if self.use_coaction:
        ph_dict = {
          "item": [self.mid_batch_ph, self.mid_his_batch_ph, self.mid_his_batch_embedded],
          "cate": [self.cate_batch_ph, self.cate_his_batch_ph, self.cate_his_batch_embedded]
        }
        self.mlp_batch_embedded = []
        with tf.device(device):
          self.item_mlp_embeddings_var = tf.get_variable("item_mlp_embedding_var",
                                                         [n_mid, INDEP_NUM * WEIGHT_EMB_DIM], trainable=True)
          self.cate_mlp_embeddings_var = tf.get_variable("cate_mlp_embedding_var",
                                                         [n_cate, INDEP_NUM * WEIGHT_EMB_DIM], trainable=True)

          self.mlp_batch_embedded.append(
            tf.nn.embedding_lookup(self.item_mlp_embeddings_var, ph_dict['item'][0]))
          self.mlp_batch_embedded.append(
            tf.nn.embedding_lookup(self.cate_mlp_embeddings_var, ph_dict['cate'][0]))

          self.input_batch_embedded = []
          self.item_input_embeddings_var = tf.get_variable("item_input_embedding_var",
                                                           [n_mid, weight_emb_w[0][0] * INDEP_NUM],
                                                           trainable=True)
          self.cate_input_embeddings_var = tf.get_variable("cate_input_embedding_var",
                                                           [n_cate, weight_emb_w[0][0] * INDEP_NUM],
                                                           trainable=True)
          self.input_batch_embedded.append(
            tf.nn.embedding_lookup(self.item_input_embeddings_var, ph_dict['item'][1]))
          self.input_batch_embedded.append(
            tf.nn.embedding_lookup(self.cate_input_embeddings_var, ph_dict['cate'][1]))

    self.item_eb = tf.concat([self.mid_batch_embedded, self.cate_batch_embedded], 1)
    self.item_his_eb = tf.concat([self.mid_his_batch_embedded, self.cate_his_batch_embedded], 2)
    self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1)
    self.item_his_eb_mean = tf.reduce_mean(self.item_his_eb, 1)
    if "FM" in self.model_type:
      self.item_eb_1dim = tf.concat([self.mid_batch_embedded_1dim, self.cate_batch_embedded_1dim], 1)
      self.item_his_eb_1dim = tf.concat([self.mid_his_batch_embedded_1dim, self.cate_his_batch_embedded_1dim], 2)
      self.item_his_eb_sum_1dim = tf.reduce_sum(self.item_his_eb_1dim, 1)
      self.item_his_eb_mean_1dim = tf.reduce_mean(self.item_his_eb_1dim, 1)

    if self.use_negsampling:
      self.noclk_item_his_eb = tf.concat(
        [self.noclk_mid_his_batch_embedded[:, :, 0, :], self.noclk_cate_his_batch_embedded[:, :, 0, :]],
        -1)  # 0 means only using the first negative item ID. 3 item IDs are inputed in the line 24.
      self.noclk_item_his_eb = tf.reshape(self.noclk_item_his_eb,
                                          [-1, tf.shape(self.noclk_mid_his_batch_embedded)[1],
                                           2 * EMBEDDING_DIM])  # cat embedding 18 concate item embedding 18.

      self.noclk_his_eb = tf.concat([self.noclk_mid_his_batch_embedded, self.noclk_cate_his_batch_embedded], -1)
      self.noclk_his_eb_sum_1 = tf.reduce_sum(self.noclk_his_eb, 2)
      self.noclk_his_eb_sum = tf.reduce_sum(self.noclk_his_eb_sum_1, 1)

    self.cross = []
    if self.use_coaction:
      input_batch = self.input_batch_embedded
      tmp_sum, tmp_seq = [], []
      if INDEP_NUM == 2:
        for i, mlp_batch in enumerate(self.mlp_batch_embedded):
          for j, input_batch in enumerate(self.input_batch_embedded):
            coaction_sum, coaction_seq = gen_coaction(
              mlp_batch[:, WEIGHT_EMB_DIM * j:  WEIGHT_EMB_DIM * (j + 1)],
              input_batch[:, :, weight_emb_w[0][0] * i: weight_emb_w[0][0] * (i + 1)], EMBEDDING_DIM,
              mode=CALC_MODE, mask=self.mask)
            tmp_sum.append(coaction_sum)
            tmp_seq.append(coaction_seq)
      else:
        for i, (mlp_batch, input_batch) in enumerate(zip(self.mlp_batch_embedded, self.input_batch_embedded)):
          coaction_sum, coaction_seq = gen_coaction(mlp_batch[:, : INDEP_NUM * WEIGHT_EMB_DIM],
                                                    input_batch[:, :, : weight_emb_w[0][0]], EMBEDDING_DIM,
                                                    mode=CALC_MODE, mask=self.mask)
          tmp_sum.append(coaction_sum)
          tmp_seq.append(coaction_seq)

      self.coaction_sum = tf.concat(tmp_sum, axis=1)
      self.cross.append(self.coaction_sum)

  def attention_din_nomask_3dims(self, cur_poi_seq_fea_col, hist_poi_seq_fea_col,
                                 din_deep_layers, din_activation, name_scope,
                                 att_type):
    # with tf.name_scope("attention_layer_%s" % (att_type)):
    # cur_poi_seq_fea_col [b, 18, eb]
    # hist_poi_seq_fea_col [b, 18, 10, eb]
    with tf.variable_scope("attention_layer_%s" % (att_type), reuse=tf.AUTO_REUSE):
      sess_num = cur_poi_seq_fea_col.get_shape().as_list()[1]  # 18 in [b,18,eb]
      embed_dim = cur_poi_seq_fea_col.get_shape().as_list()[-1]  # eb
      seq_len = hist_poi_seq_fea_col.get_shape().as_list()[-2]  # 10 in [b,18,10,eb]

      cur_poi_emb_rep = tf.tile(tf.reshape(cur_poi_seq_fea_col, [-1, sess_num, 1, embed_dim]),
                                [1, 1, seq_len, 1])  # [b,18,10,eb]
      # 将query复制 seq_len 次 None, seq_len, embed_dim
      # if att_type.startswith('top40'):
      #   din_sub = tf.subtract(cur_poi_emb_rep, hist_poi_seq_fea_col)
      #   din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col, din_sub], axis=-1)

      # elif att_type == 'click_sess_att':
      #   din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col], axis=-1)
      # elif att_type == 'order_att':
      #   din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col], axis=-1)
      # else:
      din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col], axis=-1)

      activation = tf.nn.relu if din_activation == "relu" else tf.nn.tanh
      input_layer = din_all  # [b,18,10,2*eb]
      for i in range(len(din_deep_layers)):
        deep_layer = tf.layers.dense(input_layer, int(din_deep_layers[i]), activation=activation,
                                     name=name_scope + 'f_%d_att' % (i))
        # , reuse=tf.AUTO_REUSE
        input_layer = deep_layer  # [b,18,10,32]

      din_output_layer = tf.layers.dense(input_layer, 1, activation=None,
                                         name=name_scope + 'fout_att')  # [b,18,10,1]

      weighted_outputs = din_output_layer * hist_poi_seq_fea_col
      return weighted_outputs, din_output_layer

  def attention_din_nomask(self, cur_poi_seq_fea_col, hist_poi_seq_fea_col,
                           din_deep_layers, din_activation, name_scope,
                           att_type):
    with tf.name_scope("attention_layer_%s" % (att_type)):
      # cur_poi_seq_fea_col [b, eb]
      # hist_poi_seq_fea_col [b, 10, eb]
      embed_dim = cur_poi_seq_fea_col.get_shape().as_list()[-1]  # eb
      seq_len = hist_poi_seq_fea_col.get_shape().as_list()[-2]  # 10 in [b,10,eb]
      cur_poi_emb_rep = tf.tile(tf.reshape(cur_poi_seq_fea_col, [-1, 1, embed_dim]), [1, seq_len, 1])

      # 将query复制 seq_len 次 None, seq_len, embed_dim
      if att_type.startswith('top40'):
        din_sub = tf.subtract(cur_poi_emb_rep, hist_poi_seq_fea_col)
        din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col, din_sub], axis=-1)
      elif att_type == 'click_sess_att':
        din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col], axis=-1)
      elif att_type == 'order_att':
        din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col], axis=-1)
      else:
        din_all = tf.concat([cur_poi_emb_rep, hist_poi_seq_fea_col], axis=-1)

      activation = tf.nn.relu if din_activation == "relu" else tf.nn.tanh
      input_layer = din_all
      for i in range(len(din_deep_layers)):
        deep_layer = tf.layers.dense(input_layer, int(din_deep_layers[i]), activation=activation,
                                     name=name_scope + 'f_%d_att' % (i))
        # , reuse=tf.AUTO_REUSE
        input_layer = deep_layer

      din_output_layer = tf.layers.dense(input_layer, 1, activation=None, name=name_scope + 'fout_att')  # b,10,1

      weighted_outputs = din_output_layer * hist_poi_seq_fea_col
      return weighted_outputs, din_output_layer

  def build_fcn_net(self, inp, use_dice=False):
    bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
    dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
    if use_dice:
      dnn1 = dice(dnn1, name='dice_1')
    else:
      dnn1 = prelu(dnn1, 'prelu1')

    dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
    if use_dice:
      dnn2 = dice(dnn2, name='dice_2')
    else:
      dnn2 = prelu(dnn2, 'prelu2')
    dnn3 = tf.layers.dense(dnn2, 2 if self.use_softmax else 1, activation=None, name='f3')
    return dnn3
  
  def mlp(self, inp, proj_dim, scope="mlp"):
    d = inp.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      dnn1 = tf.layers.dense(inp, d, activation=None, name='f1')
      dnn1 = prelu(dnn1, 'prelu1')

      dnn2 = tf.layers.dense(dnn1, proj_dim, activation=None, name='f2')

    return dnn2
  
  def mlp_deepmcp(self, inp, proj_dim, scope="mlp"):
    d = inp.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      dnn1 = tf.layers.dense(inp, d, activation=None, name='f1')
      dnn1 = tf.nn.relu(dnn1)

      dnn2 = tf.layers.dense(dnn1, proj_dim, activation=None, name='f2')
      dnn2 = tf.nn.tanh(dnn2)

    return dnn2


  def build_loss(self, inp):

    with tf.name_scope('Metrics'):
      # Cross-entropy loss and optimizer initialization
      if self.use_softmax:
        self.y_hat = tf.nn.softmax(inp) + 0.00000001
        ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
      else:
        self.y_hat = tf.nn.sigmoid(inp)
        ctr_loss = - tf.reduce_mean(tf.concat([tf.log(self.y_hat + 0.00000001) * self.target_ph,
                                               tf.log(1 - self.y_hat + 0.00000001) * (1 - self.target_ph)],
                                              axis=1))
      self.loss = ctr_loss

      if self.use_infonce:
        label_float_dim1 = tf.cast(tf.reshape(self.target_ph, [-1]) ,tf.float32)
        user_side_emb = tf.concat([self.uid_batch_embedded, self.self_interest, self.item_his_eb_mean], axis=1)
        poi_side_emb = self.item_eb
        
        user_proj_emb = self.mlp(user_side_emb, self.ui_proj_dim, scope="user_mlp")
        item_proj_emb = self.mlp(poi_side_emb, self.ui_proj_dim, scope="item_mlp")

        cost_ui_ui = cl.infonce_cosin_n(
          user_proj_emb, 
          item_proj_emb, 
          label_float_dim1, 
          self.ui_infonce_tau, 
          scope="ui"
        )
        cost_ui_iu = cl.infonce_cosin_n(
          item_proj_emb, 
          user_proj_emb, 
          label_float_dim1, 
          self.ui_infonce_tau, 
          scope="iu"
        )
        self.loss_infonce = 0.5*cost_ui_ui + 0.5*cost_ui_iu

        self.loss = self.loss + self.aux_infonce_w * self.loss_infonce

      if self.use_lal_infonce:
        self.loss = self.loss + self.lal_infonce_w * self.loss_lal_infonce

      self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

      # Accuracy metric
      if self.use_softmax:
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
      else:
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))

  def train(self, sess, inps):
    if self.use_infonce and self.use_lal_infonce:
      loss, loss_infonce, loss_lal_infonce, accuracy, _ = sess.run([self.loss, self.loss_infonce, self.loss_lal_infonce, self.accuracy, self.optimizer], feed_dict={
        # uids, mids, cats,
        # mid_his, cat_his, mid_mask,
        # mid_sess_his, cat_sess_his, sess_mask,
        # fin_mid_sess, fin_cat_sess,
        # target, sl, lr

        self.uid_batch_ph: inps[0],
        self.mid_batch_ph: inps[1],
        self.cate_batch_ph: inps[2],
        self.mid_his_batch_ph: inps[3],
        self.cate_his_batch_ph: inps[4],
        self.mask: inps[5],
        self.mid_sess_his: inps[6],
        self.cat_sess_his: inps[7],
        self.mid_sess_tgt: inps[8],
        self.cat_sess_tgt: inps[9],
        self.sess_mask: inps[10],
        self.fin_mid_sess: inps[11],
        self.fin_cat_sess: inps[12],
        self.target_ph: inps[13],
        self.seq_len_ph: inps[14],
        self.lr: inps[15]
      })
      return loss, loss_infonce, loss_lal_infonce, accuracy
    elif self.use_infonce:
      loss, loss_infonce, accuracy, _ = sess.run([self.loss, self.loss_infonce, self.accuracy, self.optimizer], feed_dict={
        self.uid_batch_ph: inps[0],
        self.mid_batch_ph: inps[1],
        self.cate_batch_ph: inps[2],
        self.mid_his_batch_ph: inps[3],
        self.cate_his_batch_ph: inps[4],
        self.mask: inps[5],
        self.mid_sess_his: inps[6],
        self.cat_sess_his: inps[7],
        self.mid_sess_tgt: inps[8],
        self.cat_sess_tgt: inps[9],
        self.sess_mask: inps[10],
        self.fin_mid_sess: inps[11],
        self.fin_cat_sess: inps[12],
        self.target_ph: inps[13],
        self.seq_len_ph: inps[14],
        self.lr: inps[15]
      })
      return loss, loss_infonce, accuracy
    elif self.use_lal_infonce:
      loss, loss_lal_infonce, accuracy, _ = sess.run([self.loss, self.loss_lal_infonce, self.accuracy, self.optimizer], feed_dict={
        self.uid_batch_ph: inps[0],
        self.mid_batch_ph: inps[1],
        self.cate_batch_ph: inps[2],
        self.mid_his_batch_ph: inps[3],
        self.cate_his_batch_ph: inps[4],
        self.mask: inps[5],
        self.mid_sess_his: inps[6],
        self.cat_sess_his: inps[7],
        self.mid_sess_tgt: inps[8],
        self.cat_sess_tgt: inps[9],
        self.sess_mask: inps[10],
        self.fin_mid_sess: inps[11],
        self.fin_cat_sess: inps[12],
        self.target_ph: inps[13],
        self.seq_len_ph: inps[14],
        self.lr: inps[15]
      })
      return loss, loss_lal_infonce, accuracy
    else:
      loss, accuracy, _ = sess.run([self.loss, self.accuracy, self.optimizer], feed_dict={
        self.uid_batch_ph: inps[0],
        self.mid_batch_ph: inps[1],
        self.cate_batch_ph: inps[2],
        self.mid_his_batch_ph: inps[3],
        self.cate_his_batch_ph: inps[4],
        self.mask: inps[5],
        self.mid_sess_his: inps[6],
        self.cat_sess_his: inps[7],
        self.mid_sess_tgt: inps[8],
        self.cat_sess_tgt: inps[9],
        self.sess_mask: inps[10],
        self.fin_mid_sess: inps[11],
        self.fin_cat_sess: inps[12],
        self.target_ph: inps[13],
        self.seq_len_ph: inps[14],
        self.lr: inps[15]
      })
      return loss, accuracy

  def calculate(self, sess, inps):
    if self.use_infonce and self.use_lal_infonce:
      probs, loss, loss_infonce, loss_lal_infonce, accuracy = sess.run([self.y_hat, self.loss, self.loss_infonce, self.loss_lal_infonce, self.accuracy], feed_dict={
        self.uid_batch_ph: inps[0],
        self.mid_batch_ph: inps[1],
        self.cate_batch_ph: inps[2],
        self.mid_his_batch_ph: inps[3],
        self.cate_his_batch_ph: inps[4],
        self.mask: inps[5],
        self.mid_sess_his: inps[6],
        self.cat_sess_his: inps[7],
        self.mid_sess_tgt: inps[8],
        self.cat_sess_tgt: inps[9],
        self.sess_mask: inps[10],
        self.fin_mid_sess: inps[11],
        self.fin_cat_sess: inps[12],
        self.target_ph: inps[13],
        self.seq_len_ph: inps[14]
      })
      return probs, loss, loss_infonce, loss_lal_infonce, accuracy
    elif self.use_infonce:
      probs, loss, loss_infonce, accuracy = sess.run([self.y_hat, self.loss, self.loss_infonce, self.accuracy], feed_dict={
        self.uid_batch_ph: inps[0],
        self.mid_batch_ph: inps[1],
        self.cate_batch_ph: inps[2],
        self.mid_his_batch_ph: inps[3],
        self.cate_his_batch_ph: inps[4],
        self.mask: inps[5],
        self.mid_sess_his: inps[6],
        self.cat_sess_his: inps[7],
        self.mid_sess_tgt: inps[8],
        self.cat_sess_tgt: inps[9],
        self.sess_mask: inps[10],
        self.fin_mid_sess: inps[11],
        self.fin_cat_sess: inps[12],
        self.target_ph: inps[13],
        self.seq_len_ph: inps[14]
      })
      return probs, loss, loss_infonce, accuracy
    elif self.use_lal_infonce:
      probs, loss, loss_lal_infonce, accuracy = sess.run([self.y_hat, self.loss, self.loss_lal_infonce, self.accuracy], feed_dict={
        self.uid_batch_ph: inps[0],
        self.mid_batch_ph: inps[1],
        self.cate_batch_ph: inps[2],
        self.mid_his_batch_ph: inps[3],
        self.cate_his_batch_ph: inps[4],
        self.mask: inps[5],
        self.mid_sess_his: inps[6],
        self.cat_sess_his: inps[7],
        self.mid_sess_tgt: inps[8],
        self.cat_sess_tgt: inps[9],
        self.sess_mask: inps[10],
        self.fin_mid_sess: inps[11],
        self.fin_cat_sess: inps[12],
        self.target_ph: inps[13],
        self.seq_len_ph: inps[14]
      })
      return probs, loss, loss_lal_infonce, accuracy
    else:
      probs, loss, accuracy = sess.run([self.y_hat, self.loss, self.accuracy], feed_dict={
        self.uid_batch_ph: inps[0],
        self.mid_batch_ph: inps[1],
        self.cate_batch_ph: inps[2],
        self.mid_his_batch_ph: inps[3],
        self.cate_his_batch_ph: inps[4],
        self.mask: inps[5],
        self.mid_sess_his: inps[6],
        self.cat_sess_his: inps[7],
        self.mid_sess_tgt: inps[8],
        self.cat_sess_tgt: inps[9],
        self.sess_mask: inps[10],
        self.fin_mid_sess: inps[11],
        self.fin_cat_sess: inps[12],
        self.target_ph: inps[13],
        self.seq_len_ph: inps[14]
      })
      return probs, loss, accuracy

  def save(self, sess, path):
    saver = tf.train.Saver()
    saver.save(sess, save_path=path)

  def restore(self, sess, path):
    saver = tf.train.Saver()
    saver.restore(sess, save_path=path)
    print('model restored from %s' % path)

class Model_DNN(Model):
  def __init__(self, n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False,
               use_softmax=True, use_infonce=False, ui_infonce_tau=0.07, aux_infonce_w=0.05, ui_proj_dim=16,
               use_lal_infonce=False, lal_infonce_tau=0.07, lal_infonce_w=0.01,infonce_loss_type="contrastive",
               model_type="DBPMaN"):
    super(Model_DNN, self).__init__(n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE,
                                    ATTENTION_SIZE,
                                    use_negsampling, use_softmax=use_softmax,
                                    use_infonce=use_infonce, ui_infonce_tau=ui_infonce_tau, 
                                    aux_infonce_w=aux_infonce_w, ui_proj_dim=ui_proj_dim,
                                    use_lal_infonce=use_lal_infonce, lal_infonce_tau=lal_infonce_tau,
                                    lal_infonce_w=lal_infonce_w, infonce_loss_type=infonce_loss_type,
                                    model_type=model_type
                                  )
                                  
    if self.use_infonce:
      self.self_interest = self_agg(self.item_his_eb, self.mask, scope="self_agg")
      inp = tf.concat(
        [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, self.self_interest],
        -1)
    else:
      inp = tf.concat(
        [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum],
        -1)
    # Fully connected layer
    logit = self.build_fcn_net(inp, use_dice=True)
    self.build_loss(logit)

class Model_FM(Model):
  def __init__(self, n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False,
               use_softmax=True, use_infonce=False, ui_infonce_tau=0.07, aux_infonce_w=0.05, ui_proj_dim=16,
               use_lal_infonce=False, lal_infonce_tau=0.07, lal_infonce_w=0.01,infonce_loss_type="contrastive",
               model_type="DBPMaN"):
    super(Model_FM, self).__init__(n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE,
                                    ATTENTION_SIZE,
                                    use_negsampling, use_softmax=use_softmax,
                                    use_infonce=use_infonce, ui_infonce_tau=ui_infonce_tau, 
                                    aux_infonce_w=aux_infonce_w, ui_proj_dim=ui_proj_dim,
                                    use_lal_infonce=use_lal_infonce, lal_infonce_tau=lal_infonce_tau,
                                    lal_infonce_w=lal_infonce_w, infonce_loss_type=infonce_loss_type,
                                    model_type=model_type
                                  )
    all_emb_1dim = tf.concat([
      self.uid_batch_embedded_1dim,
      self.item_eb_1dim,
      self.item_his_eb_mean_1dim
    ], axis=1)

    if self.use_infonce:
      self.self_interest = self_agg(self.item_his_eb, self.mask, scope="self_agg")
      all_emb_2dim = tf.concat([
        self.uid_batch_embedded,
        self.item_eb,
        self.item_his_eb_mean,
        self.self_interest
      ], axis=1)
    else:
      all_emb_2dim = tf.concat([
        self.uid_batch_embedded,
        self.item_eb,
        self.item_his_eb_mean
      ], axis=1)

    bs = tf.shape(all_emb_2dim)[0]
    all_emb_3dim = tf.reshape(all_emb_2dim, [bs, -1, 18]) # b*t*
    
    square_of_sum = tf.square(tf.reduce_sum(all_emb_3dim, axis=1, keep_dims=True))
    sum_of_square = tf.reduce_sum(all_emb_3dim*all_emb_3dim, axis=1, keep_dims=True)
    cross_term = square_of_sum - sum_of_square
    logits_2nd_order = 0.5 * tf.reduce_sum(cross_term, axis=2, keep_dims=False)
    logits_1st_order = tf.reduce_sum(all_emb_1dim, axis=1, keep_dims=True)
    logit = logits_1st_order + logits_2nd_order

    self.build_loss(logit)

class Model_DeepFM(Model):
  def __init__(self, n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False,
               use_softmax=True, use_infonce=False, ui_infonce_tau=0.07, aux_infonce_w=0.05, ui_proj_dim=16,
               use_lal_infonce=False, lal_infonce_tau=0.07, lal_infonce_w=0.01,infonce_loss_type="contrastive",
               model_type="DBPMaN"):
    super(Model_DeepFM, self).__init__(n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE,
                                    ATTENTION_SIZE,
                                    use_negsampling, use_softmax=use_softmax,
                                    use_infonce=use_infonce, ui_infonce_tau=ui_infonce_tau, 
                                    aux_infonce_w=aux_infonce_w, ui_proj_dim=ui_proj_dim,
                                    use_lal_infonce=use_lal_infonce, lal_infonce_tau=lal_infonce_tau,
                                    lal_infonce_w=lal_infonce_w, infonce_loss_type=infonce_loss_type,
                                    model_type=model_type
                                  )
                                  
    all_emb_1dim = tf.concat([
      self.uid_batch_embedded_1dim,
      self.item_eb_1dim,
      self.item_his_eb_mean_1dim
    ], axis=1)

    if self.use_infonce:
      self.self_interest = self_agg(self.item_his_eb, self.mask, scope="self_agg")
      all_emb_2dim = tf.concat([
        self.uid_batch_embedded,
        self.item_eb,
        self.item_his_eb_mean,
        self.self_interest
      ], axis=1)
    else:
      all_emb_2dim = tf.concat([
        self.uid_batch_embedded,
        self.item_eb,
        self.item_his_eb_mean
      ], axis=1)

    bs = tf.shape(all_emb_2dim)[0]
    all_emb_3dim = tf.reshape(all_emb_2dim, [bs, -1, 18]) # b*t*d

    square_of_sum = tf.square(tf.reduce_sum(all_emb_3dim, axis=1, keep_dims=True))
    sum_of_square = tf.reduce_sum(all_emb_3dim*all_emb_3dim, axis=1, keep_dims=True)
    cross_term = square_of_sum - sum_of_square
    logits_2nd_order = 0.5 * tf.reduce_sum(cross_term, axis=2, keep_dims=False)
    logits_1st_order = tf.reduce_sum(all_emb_1dim, axis=1, keep_dims=True)

    logits_dnn = self.build_fcn_net(all_emb_2dim, use_dice=True)

    logit = logits_1st_order + logits_2nd_order + logits_dnn

    self.build_loss(logit)

class Model_WideDeep(Model):
  def __init__(self, n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False,
               use_softmax=True, use_infonce=False, ui_infonce_tau=0.07, aux_infonce_w=0.05, ui_proj_dim=16,
               use_lal_infonce=False, lal_infonce_tau=0.07, lal_infonce_w=0.01,infonce_loss_type="contrastive",
               model_type="DBPMaN"):
    super(Model_WideDeep, self).__init__(n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE,
                                    ATTENTION_SIZE,
                                    use_negsampling, use_softmax=use_softmax,
                                    use_infonce=use_infonce, ui_infonce_tau=ui_infonce_tau, 
                                    aux_infonce_w=aux_infonce_w, ui_proj_dim=ui_proj_dim,
                                    use_lal_infonce=use_lal_infonce, lal_infonce_tau=lal_infonce_tau,
                                    lal_infonce_w=lal_infonce_w, infonce_loss_type=infonce_loss_type,
                                    model_type=model_type
                                  )
                                  
    if self.use_infonce:
      self.self_interest = self_agg(self.item_his_eb, self.mask, scope="self_agg")
      inp = tf.concat(
        [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, self.self_interest],
        -1)
    else:
      inp = tf.concat(
        [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum],
        -1)
    # Fully connected layer
    logit = self.build_fcn_net(inp, use_dice=True)

    logit_wide = tf.layers.dense(inp, 1, activation=None, name="fc_wide")

    self.build_loss(logit + logit_wide)

class Model_DIN(Model):
  def __init__(self, n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False,
               use_softmax=True, use_infonce=False, ui_infonce_tau=0.07, aux_infonce_w=0.05, ui_proj_dim=16,
               use_lal_infonce=False, lal_infonce_tau=0.07, lal_infonce_w=0.01,infonce_loss_type="contrastive",
               model_type="DBPMaN"):
    super(Model_DIN, self).__init__(n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE,
                                    ATTENTION_SIZE,
                                    use_negsampling, use_softmax=use_softmax,
                                    use_infonce=use_infonce, ui_infonce_tau=ui_infonce_tau, 
                                    aux_infonce_w=aux_infonce_w, ui_proj_dim=ui_proj_dim,
                                    use_lal_infonce=use_lal_infonce, lal_infonce_tau=lal_infonce_tau,
                                    lal_infonce_w=lal_infonce_w, infonce_loss_type=infonce_loss_type,
                                    model_type=model_type
                                  )
                                  

    # Attention layer
    with tf.variable_scope('DIN'):
      attention_output = din_attention(self.item_eb, self.item_his_eb, ATTENTION_SIZE, self.mask)
      att_fea = tf.reduce_sum(attention_output, 1)
      if self.use_infonce:
        self.self_interest = self_agg(self.item_his_eb, self.mask, scope="self_agg")
        inp = tf.concat(
          [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, att_fea, self.self_interest],
          -1)
      else:
        inp = tf.concat(
          [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, att_fea],
          -1)
    # Fully connected layer
    logit = self.build_fcn_net(inp, use_dice=True)
    self.build_loss(logit)

class Model_DIEN(Model):
  def __init__(self, n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False,
               use_coaction=False, use_infonce=False, ui_infonce_tau=0.07, aux_infonce_w=0.05, ui_proj_dim=16,
               use_lal_infonce=False, lal_infonce_tau=0.07, lal_infonce_w=0.01,infonce_loss_type="contrastive",
               model_type="DBPMaN"):
    super(Model_DIEN, self).__init__(n_uid, n_mid, n_cate,
                                     EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                     use_negsampling, use_coaction=use_coaction,
                                     use_infonce=use_infonce, ui_infonce_tau=ui_infonce_tau, 
                                     aux_infonce_w=aux_infonce_w, ui_proj_dim=ui_proj_dim,
                                     use_lal_infonce=use_lal_infonce, lal_infonce_tau=lal_infonce_tau,
                                     lal_infonce_w=lal_infonce_w, infonce_loss_type=infonce_loss_type,
                                     model_type=model_type
                                    )
                                    
    # RNN layer(-s)
    with tf.name_scope('rnn_1'):
      rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                   sequence_length=self.seq_len_ph, dtype=tf.float32,
                                   scope="gru1")

    # Attention layer
    with tf.name_scope('Attention_layer_1'):
      att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE, self.mask,
                                              softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)

    with tf.name_scope('rnn_2'):
      rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                               att_scores=tf.expand_dims(alphas, -1),
                                               sequence_length=self.seq_len_ph, dtype=tf.float32,
                                               scope="gru2")
    if self.use_infonce:
      self.self_interest = self_agg(self.item_his_eb, self.mask, scope="self_agg")
      inp = tf.concat(
        [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum,
        final_state2, self.self_interest] + self.cross, 1)    
    else:
      inp = tf.concat(
        [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum,
        final_state2] + self.cross, 1)
    prop = self.build_fcn_net(inp, use_dice=True)
    self.build_loss(prop)

class Model_DBPMaN(Model):
  def __init__(self, n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False,
               use_softmax=True, use_infonce=False, ui_infonce_tau=0.07, aux_infonce_w=0.05, ui_proj_dim=16,
               use_lal_infonce=False, lal_infonce_tau=0.07, lal_infonce_w=0.01,infonce_loss_type="contrastive",
               model_type="DBPMaN"):
    super(Model_DBPMaN, self).__init__(n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE,
                                       ATTENTION_SIZE,
                                       use_negsampling, use_softmax=use_softmax,
                                       use_infonce=use_infonce, ui_infonce_tau=ui_infonce_tau, 
                                       aux_infonce_w=aux_infonce_w, ui_proj_dim=ui_proj_dim,
                                       use_lal_infonce=use_lal_infonce, lal_infonce_tau=lal_infonce_tau,
                                       lal_infonce_w=lal_infonce_w, infonce_loss_type=infonce_loss_type,
                                       model_type=model_type
                                      )
                                      
    self.mid_sess_his_eb = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_sess_his)  # [1024, 18, 10, eb]
    self.cat_sess_his_eb = tf.nn.embedding_lookup(self.cate_embeddings_var, self.cat_sess_his)
    self.mid_sess_his_eb += self.cat_sess_his_eb
    self.mid_sess_tgt_eb = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_sess_tgt)  # [1024, 18, eb]
    self.cat_sess_tgt_eb = tf.nn.embedding_lookup(self.cate_embeddings_var, self.cat_sess_tgt)
    self.mid_sess_tgt_eb += self.cat_sess_tgt_eb
    self.fin_mid_sess_eb = tf.nn.embedding_lookup(self.mid_embeddings_var, self.fin_mid_sess)  # [1024, 10, eb]
    self.fin_cat_sess_eb = tf.nn.embedding_lookup(self.cate_embeddings_var, self.fin_cat_sess)
    self.fin_mid_sess_eb += self.fin_cat_sess_eb

    # Attention layer
    with tf.variable_scope('DBPMaN_Model', reuse=tf.AUTO_REUSE):
      # 1. Pathway Enhance Module
      # 1.1 history path enhance
      # [b, 18, 10 ,eb]
      mid_sess_his_eb_enhance, nclk_his_att_score = self.attention_din_nomask_3dims(
        tf.stop_gradient(self.mid_sess_tgt_eb),
        self.mid_sess_his_eb,
        [64, 32], 
        'relu',
        'pem_his_att',
        'din_mlp'
      )

      mask_pw_his_router = self.pathway_router_simple(mid_sess_his_eb_enhance, [32], 10,
                                                      'his_router')  # b,18,10,1

      mid_sess_his_eb_enhance = tf.reduce_sum((mid_sess_his_eb_enhance * mask_pw_his_router), axis=-2)  # b,18,eb
      out_fea_0 = tf.reduce_mean(mid_sess_his_eb_enhance, axis=-2)

      if self.use_lal_infonce:
        user_proj_emb = mid_sess_his_eb_enhance # b,18,eb
        item_proj_emb = self.mid_sess_tgt_eb    # b,18,eb

        cost_ui = cl.infonce_cosin_n(
          user_proj_emb,
          item_proj_emb,
          tf.cast(self.sess_mask, tf.float32),
          self.lal_infonce_tau,
          scope="ui"
        )

        cost_iu = cl.infonce_cosin_n(
          item_proj_emb,
          user_proj_emb,
          tf.cast(self.sess_mask, tf.float32),
          self.lal_infonce_tau,
          scope="iu"
        )

        self.loss_lal_infonce = 0.5 * cost_ui + 0.5 * cost_iu

      mid_sess_cur_eb_enhance, nclk_cur_att_score = self.attention_din_nomask(
        tf.stop_gradient(self.mid_batch_embedded),
        self.fin_mid_sess_eb,
        [64, 32], 
        'relu',
        'pem_cur_att', 
        'din_mlp'
      )

      mask_pw_cur_router = self.pathway_router_simple(mid_sess_cur_eb_enhance, [32], 10,
                                                      'cur_router')  # b,10,1
      mid_sess_cur_eb_enhance = tf.reduce_sum((mid_sess_cur_eb_enhance * mask_pw_cur_router), axis=-2)  # b,eb
      out_fea_1 = mid_sess_cur_eb_enhance

      # 2. Pathway Matching Module
      mid_sess_cur_eb_pool = tf.reshape(mid_sess_cur_eb_enhance, [-1, 1, EMBEDDING_DIM])  # [B, 1, eb]
      mid_sess_his_eb_pool = mid_sess_his_eb_enhance  # [B, 18, eb]

      sess_score = tf.matmul(mid_sess_cur_eb_pool, tf.transpose(mid_sess_his_eb_pool, [0, 2, 1]))  # [B, 1, 18]
      attention_output_0 = din_attention(tf.squeeze(mid_sess_cur_eb_pool, 1), mid_sess_his_eb_pool,
                                         ATTENTION_SIZE, self.sess_mask, name_scope='attention_output_0')

      att_fea_0 = tf.reduce_sum(attention_output_0, 1)
      attention_output_1 = din_attention(self.item_eb, self.item_his_eb, ATTENTION_SIZE, self.mask,
                                         att_score=tf.squeeze(sess_score, 1), name_scope='attention_output1')
      att_fea_1 = tf.reduce_sum(attention_output_1, 1)

    if self.use_infonce:
      self.self_interest = self_agg(self.item_his_eb, self.mask, scope="self_agg")
      inp = tf.concat(
        [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum,
        att_fea_0, att_fea_1, out_fea_0, out_fea_1, self.self_interest], -1)
    else:
      inp = tf.concat(
        [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum,
        att_fea_0, att_fea_1, out_fea_0, out_fea_1], -1)

    # Fully connected layer
    logit = self.build_fcn_net(inp, use_dice=True)
    self.build_loss(logit)

  def pathway_router_simple(self, input_layer, hidden_units_list, output_dim, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
      net = input_layer  # b,18,10,eb or b,10,eb

      # net_reshape = tf.reduce_mean(net, axis=-1)  # b,18,10
      net_reshape = tf.squeeze(tf.concat(tf.split(net, net.shape[-2].value, axis=-2), axis=-1),
                               axis=-2)  # b,18,10*eb
      # second loop
      for i in range(len(hidden_units_list)):
        net_reshape = tf.layers.dense(inputs=net_reshape,
                                      units=hidden_units_list[i],
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.glorot_normal_initializer(),
                                      bias_initializer=tf.glorot_normal_initializer(),
                                      name='%s_fc_pw_%d' % (name, i))

      pw_router = tf.layers.dense(inputs=net_reshape,
                                  units=output_dim,
                                  activation=tf.nn.softmax,
                                  # activation=tf.nn.sigmoid,
                                  kernel_initializer=tf.glorot_normal_initializer(),
                                  bias_initializer=tf.glorot_normal_initializer(),
                                  name='%s_softmax_pw' % (name))  # b,18,10

      topk_vals, _ = tf.nn.top_k(pw_router, 5)  # B,18,5
      min_topk_vals = tf.reduce_min(topk_vals, axis=-1, keepdims=True)  # B,18,1

      pw_bool = tf.math.greater_equal(pw_router, min_topk_vals)  # B, 18, 10
      mask_val = tf.zeros_like(pw_router)
      mask_pw_router = tf.where(pw_bool, pw_router, mask_val)  # b,18,10

    return tf.expand_dims(mask_pw_router, axis=-1)  # b,18,10,1

class Model_DeepMCP(Model):
  def __init__(self, n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False,
               use_softmax=True, use_infonce=False, ui_infonce_tau=0.07, aux_infonce_w=0.05, ui_proj_dim=16,
               use_lal_infonce=False, lal_infonce_tau=0.07, lal_infonce_w=0.01,infonce_loss_type="contrastive",
               model_type="DBPMaN"):
    super(Model_DeepMCP, self).__init__(n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE,
                                       ATTENTION_SIZE,
                                       use_negsampling, use_softmax=use_softmax,
                                       use_infonce=use_infonce, ui_infonce_tau=ui_infonce_tau, 
                                       aux_infonce_w=aux_infonce_w, ui_proj_dim=ui_proj_dim,
                                       use_lal_infonce=use_lal_infonce, lal_infonce_tau=lal_infonce_tau,
                                       lal_infonce_w=lal_infonce_w, infonce_loss_type=infonce_loss_type,
                                       model_type=model_type
                                      )

    user_proj_emb = self.mlp_deepmcp(self.uid_batch_embedded, self.ui_proj_dim, scope="user_mlp")
    item_proj_emb = self.mlp_deepmcp(self.item_eb, self.ui_proj_dim, scope="item_mlp")

    match_logit = tf.reduce_sum(tf.multiply(user_proj_emb, item_proj_emb), 1, keep_dims=True)
    match_hat = tf.nn.sigmoid(match_logit)
    self.loss_lal_infonce = -tf.reduce_mean(tf.concat([tf.log(match_hat + 0.00000001) * self.target_ph,
                                          tf.log(1 - match_hat + 0.00000001) * (1 - self.target_ph)],
                                        axis=1))

    self.mid_sess_his_eb = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_sess_his)  # [1024, 18, 10, eb]
    self.cat_sess_his_eb = tf.nn.embedding_lookup(self.cate_embeddings_var, self.cat_sess_his)
    self.mid_sess_his_eb += self.cat_sess_his_eb
    self.mid_sess_tgt_eb = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_sess_tgt)  # [1024, 18, eb]
    self.cat_sess_tgt_eb = tf.nn.embedding_lookup(self.cate_embeddings_var, self.cat_sess_tgt)
    self.mid_sess_tgt_eb += self.cat_sess_tgt_eb
    self.fin_mid_sess_eb = tf.nn.embedding_lookup(self.mid_embeddings_var, self.fin_mid_sess)  # [1024, 10, eb]
    self.fin_cat_sess_eb = tf.nn.embedding_lookup(self.cate_embeddings_var, self.fin_cat_sess)
    self.fin_mid_sess_eb += self.fin_cat_sess_eb

    # Attention layer
    with tf.variable_scope('DBPMaN_Model', reuse=tf.AUTO_REUSE):
      # 1. Pathway Enhance Module
      # 1.1 history path enhance
      # [b, 18, 10 ,eb]
      mid_sess_his_eb_enhance, nclk_his_att_score = self.attention_din_nomask_3dims(
        tf.stop_gradient(self.mid_sess_tgt_eb),
        self.mid_sess_his_eb,
        [64, 32], 
        'relu',
        'pem_his_att',
        'din_mlp'
      )

      mask_pw_his_router = self.pathway_router_simple(mid_sess_his_eb_enhance, [32], 10,
                                                      'his_router')  # b,18,10,1

      mid_sess_his_eb_enhance = tf.reduce_sum((mid_sess_his_eb_enhance * mask_pw_his_router), axis=-2)  # b,18,eb
      out_fea_0 = tf.reduce_mean(mid_sess_his_eb_enhance, axis=-2)

      mid_sess_cur_eb_enhance, nclk_cur_att_score = self.attention_din_nomask(
        tf.stop_gradient(self.mid_batch_embedded),
        self.fin_mid_sess_eb,
        [64, 32], 
        'relu',
        'pem_cur_att', 
        'din_mlp'
      )

      mask_pw_cur_router = self.pathway_router_simple(mid_sess_cur_eb_enhance, [32], 10,
                                                      'cur_router')  # b,10,1
      mid_sess_cur_eb_enhance = tf.reduce_sum((mid_sess_cur_eb_enhance * mask_pw_cur_router), axis=-2)  # b,eb
      out_fea_1 = mid_sess_cur_eb_enhance

      # 2. Pathway Matching Module
      mid_sess_cur_eb_pool = tf.reshape(mid_sess_cur_eb_enhance, [-1, 1, EMBEDDING_DIM])  # [B, 1, eb]
      mid_sess_his_eb_pool = mid_sess_his_eb_enhance  # [B, 18, eb]

      sess_score = tf.matmul(mid_sess_cur_eb_pool, tf.transpose(mid_sess_his_eb_pool, [0, 2, 1]))  # [B, 1, 18]
      attention_output_0 = din_attention(tf.squeeze(mid_sess_cur_eb_pool, 1), mid_sess_his_eb_pool,
                                         ATTENTION_SIZE, self.sess_mask, name_scope='attention_output_0')

      att_fea_0 = tf.reduce_sum(attention_output_0, 1)
      attention_output_1 = din_attention(self.item_eb, self.item_his_eb, ATTENTION_SIZE, self.mask,
                                         att_score=tf.squeeze(sess_score, 1), name_scope='attention_output1')
      att_fea_1 = tf.reduce_sum(attention_output_1, 1)

    inp = tf.concat(
      [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum,
      att_fea_0, att_fea_1, out_fea_0, out_fea_1], -1)

    # Fully connected layer
    logit = self.build_fcn_net(inp, use_dice=True)
    self.build_loss(logit)

  def pathway_router_simple(self, input_layer, hidden_units_list, output_dim, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
      net = input_layer  # b,18,10,eb or b,10,eb

      # net_reshape = tf.reduce_mean(net, axis=-1)  # b,18,10
      net_reshape = tf.squeeze(tf.concat(tf.split(net, net.shape[-2].value, axis=-2), axis=-1),
                               axis=-2)  # b,18,10*eb
      # second loop
      for i in range(len(hidden_units_list)):
        net_reshape = tf.layers.dense(inputs=net_reshape,
                                      units=hidden_units_list[i],
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.glorot_normal_initializer(),
                                      bias_initializer=tf.glorot_normal_initializer(),
                                      name='%s_fc_pw_%d' % (name, i))

      pw_router = tf.layers.dense(inputs=net_reshape,
                                  units=output_dim,
                                  activation=tf.nn.softmax,
                                  # activation=tf.nn.sigmoid,
                                  kernel_initializer=tf.glorot_normal_initializer(),
                                  bias_initializer=tf.glorot_normal_initializer(),
                                  name='%s_softmax_pw' % (name))  # b,18,10

      topk_vals, _ = tf.nn.top_k(pw_router, 5)  # B,18,5
      min_topk_vals = tf.reduce_min(topk_vals, axis=-1, keepdims=True)  # B,18,1

      pw_bool = tf.math.greater_equal(pw_router, min_topk_vals)  # B, 18, 10
      mask_val = tf.zeros_like(pw_router)
      mask_pw_router = tf.where(pw_bool, pw_router, mask_val)  # b,18,10

    return tf.expand_dims(mask_pw_router, axis=-1)  # b,18,10,1

class Model_DMR(Model):
  def __init__(self, n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False,
               use_softmax=True, use_infonce=False, ui_infonce_tau=0.07, aux_infonce_w=0.05, ui_proj_dim=16,
               use_lal_infonce=False, lal_infonce_tau=0.07, lal_infonce_w=0.01,infonce_loss_type="contrastive",
               model_type="DBPMaN"):
    super(Model_DMR, self).__init__(n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE,
                                    ATTENTION_SIZE,
                                    use_negsampling, use_softmax=use_softmax,
                                    use_infonce=use_infonce, ui_infonce_tau=ui_infonce_tau, 
                                    aux_infonce_w=aux_infonce_w, ui_proj_dim=ui_proj_dim,
                                    use_lal_infonce=use_lal_infonce, lal_infonce_tau=lal_infonce_tau,
                                    lal_infonce_w=lal_infonce_w, infonce_loss_type=infonce_loss_type,
                                    model_type=model_type
                                  )
    # Attention layer
    with tf.variable_scope('DIN'):
      dm_item_vectors = tf.get_variable("dm_item_vectors", [n_mid, EMBEDDING_DIM])
      dm_item_bias = tf.get_variable("dm_item_bias", [n_mid], initializer=tf.zeros_initializer(), trainable=False)

      self.loss_lal_infonce, dm_user_vector = self.deep_match(
        self.item_his_eb,
        self.mask,
        self.mid_his_batch_ph,
        EMBEDDING_DIM,
        dm_item_vectors,
        dm_item_bias,
        n_mid
      )
      dm_item_vec = tf.nn.embedding_lookup(dm_item_vectors, self.mid_batch_ph)
      rel_u2i = tf.reduce_sum(dm_user_vector * dm_item_vec, axis=-1, keep_dims=True)

      attention_output = din_attention(self.item_eb, self.item_his_eb, ATTENTION_SIZE, self.mask)
      att_fea = tf.reduce_sum(attention_output, 1)
      inp = tf.concat(
        [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, rel_u2i, att_fea],
        -1)
    # Fully connected layer
    logit = self.build_fcn_net(inp, use_dice=True)
    self.build_loss(logit)
  
  def deep_match(self, item_his_emb, mask, item_his_batch, EMBEDDING_DIM, item_vectors, item_bias, n_item):
    with tf.variable_scope("deep_match", reuse=tf.AUTO_REUSE):
      inputs = item_his_emb
      att_layer1 = tf.layers.dense(inputs, 80, activation=tf.nn.sigmoid, name='dm_att_1')
      att_layer2 = tf.layers.dense(att_layer1, 40, activation=tf.nn.sigmoid, name='dm_att_2')
      att_layer3 = tf.layers.dense(att_layer2, 1, activation=None, name='dm_att_3')  # B,T,1
      scores = tf.transpose(att_layer3, [0, 2, 1]) # B,1,T

      bool_mask = tf.equal(mask, tf.ones_like(mask))  # B,T
      key_masks = tf.expand_dims(bool_mask, 1)  # B,1,T
      paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
      scores = tf.where(key_masks, scores, paddings)

      scores_tile = tf.tile(tf.reduce_sum(scores, axis=1), [1, tf.shape(scores)[-1]]) # B, T*T
      scores_tile = tf.reshape(scores_tile, [-1, tf.shape(scores)[-1], tf.shape(scores)[-1]]) # B, T, T
      diag_vals = tf.ones_like(scores_tile)  # B, T, T

      tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
      paddings = tf.ones_like(tril) * (-2 ** 32 + 1)
      scores_tile = tf.where(tf.equal(tril, 0), paddings, scores_tile)  # B, T, T
      scores_tile = tf.nn.softmax(scores_tile) # B, T, T
      att_dm_item_his_eb = tf.matmul(scores_tile, item_his_emb) # B, T, E

      dnn_layer1 = tf.layers.dense(att_dm_item_his_eb, EMBEDDING_DIM, activation=None, name='dm_fcn_1')
      dnn_layer1 = prelu(dnn_layer1, 'dm_fcn_1') # B, T, E

      user_vector = dnn_layer1[:, -1, :]

      user_vector2 = dnn_layer1[:, -2, :]

      num_sampled = 2000

      loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=item_vectors,
                                                      biases=item_bias,
                                                      labels=tf.cast(tf.reshape(item_his_batch[:, -1], [-1,1]), tf.int64),
                                                      inputs=user_vector2,
                                                      num_sampled=num_sampled,
                                                      num_classes=n_item,
                                                      sampled_values=tf.nn.learned_unigram_candidate_sampler(tf.cast(tf.reshape(item_his_batch[:, -1], [-1, 1]), tf.int64), 1, num_sampled, True, n_item)
                                                      ))
    return loss, user_vector
  
class Model_CL4CTR(Model):
  def __init__(self, n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False,
               use_softmax=True, use_infonce=False, ui_infonce_tau=0.07, aux_infonce_w=0.05, ui_proj_dim=16,
               use_lal_infonce=False, lal_infonce_tau=0.07, lal_infonce_w=0.01,infonce_loss_type="contrastive",
               model_type="DBPMaN"):
    super(Model_CL4CTR, self).__init__(n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE,
                                       ATTENTION_SIZE,
                                       use_negsampling, use_softmax=use_softmax,
                                       use_infonce=use_infonce, ui_infonce_tau=ui_infonce_tau, 
                                       aux_infonce_w=aux_infonce_w, ui_proj_dim=ui_proj_dim,
                                       use_lal_infonce=use_lal_infonce, lal_infonce_tau=lal_infonce_tau,
                                       lal_infonce_w=lal_infonce_w, infonce_loss_type=infonce_loss_type,
                                       model_type=model_type
                                      )
                                      
    self.mid_sess_his_eb = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_sess_his)  # [1024, 18, 10, eb]
    self.cat_sess_his_eb = tf.nn.embedding_lookup(self.cate_embeddings_var, self.cat_sess_his)
    self.mid_sess_his_eb += self.cat_sess_his_eb
    self.mid_sess_tgt_eb = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_sess_tgt)  # [1024, 18, eb]
    self.cat_sess_tgt_eb = tf.nn.embedding_lookup(self.cate_embeddings_var, self.cat_sess_tgt)
    self.mid_sess_tgt_eb += self.cat_sess_tgt_eb
    self.fin_mid_sess_eb = tf.nn.embedding_lookup(self.mid_embeddings_var, self.fin_mid_sess)  # [1024, 10, eb]
    self.fin_cat_sess_eb = tf.nn.embedding_lookup(self.cate_embeddings_var, self.fin_cat_sess)
    self.fin_mid_sess_eb += self.fin_cat_sess_eb

    all_emb_2dim = tf.concat([
      self.uid_batch_embedded,
      self.item_eb,
      self.item_his_eb_mean
    ], axis=1)

    bs = tf.shape(all_emb_2dim)[0]
    all_emb_3dim = tf.reshape(all_emb_2dim, [bs, -1, EMBEDDING_DIM]) # b*t*d
    tmp_mask = tf.ones(shape=[bs, tf.shape(all_emb_3dim)[1]], dtype=tf.float32)

    with tf.variable_scope("cl4ctr", reuse=tf.AUTO_REUSE):
      all_emb_view1_3dim = tf.layers.dropout(all_emb_3dim, 0.2, training=True)
      all_emb_view2_3dim = tf.layers.dropout(all_emb_3dim, 0.2, training=True)

      all_emb_view1_3dim_out1 = transformer.encoder(all_emb_view1_3dim, tmp_mask, 1, 0.2, 72, True, False, scope="encoder_1")
      all_emb_view2_3dim_out1 = transformer.encoder(all_emb_view2_3dim, tmp_mask, 1, 0.2, 72, True, False, scope="encoder_1")

      all_emb_view1_3dim_out2 = transformer.encoder(all_emb_view1_3dim_out1, tmp_mask, 1, 0.2, 72, True, False, scope="encoder_2")
      all_emb_view2_3dim_out2 = transformer.encoder(all_emb_view2_3dim_out1, tmp_mask, 1, 0.2, 72, True, False, scope="encoder_2")

      all_emb_view1_3dim_out3 = transformer.encoder(all_emb_view1_3dim_out2, tmp_mask, 1, 0.2, 72, True, False, scope="encoder_3")
      all_emb_view2_3dim_out3 = transformer.encoder(all_emb_view2_3dim_out2, tmp_mask, 1, 0.2, 72, True, False, scope="encoder_3")

      all_emb_view1_3dim_out3 = tf.reshape(all_emb_view1_3dim_out3, [bs,5*18])
      all_emb_view2_3dim_out3 = tf.reshape(all_emb_view2_3dim_out3, [bs,5*18])

      view1_proj = tf.layers.dense(
        all_emb_view1_3dim_out3,
        32,
        activation=None,
        kernel_initializer=tf.initializers.truncated_normal(0, 0.05)
      )

      view2_proj = tf.layers.dense(
        all_emb_view2_3dim_out3,
        32,
        activation=None,
        kernel_initializer=tf.initializers.truncated_normal(0, 0.05)
      )

      self.loss_lal_infonce = tf.reduce_mean(
        tf.pow((view1_proj - view2_proj), 2.0)
      )

    # Attention layer
    with tf.variable_scope('DBPMaN_Model', reuse=tf.AUTO_REUSE):
      # 1. Pathway Enhance Module
      # 1.1 history path enhance
      # [b, 18, 10 ,eb]
      mid_sess_his_eb_enhance, nclk_his_att_score = self.attention_din_nomask_3dims(
        tf.stop_gradient(self.mid_sess_tgt_eb),
        self.mid_sess_his_eb,
        [64, 32], 
        'relu',
        'pem_his_att',
        'din_mlp'
      )

      mask_pw_his_router = self.pathway_router_simple(mid_sess_his_eb_enhance, [32], 10,
                                                      'his_router')  # b,18,10,1

      mid_sess_his_eb_enhance = tf.reduce_sum((mid_sess_his_eb_enhance * mask_pw_his_router), axis=-2)  # b,18,eb
      out_fea_0 = tf.reduce_mean(mid_sess_his_eb_enhance, axis=-2)

      mid_sess_cur_eb_enhance, nclk_cur_att_score = self.attention_din_nomask(
        tf.stop_gradient(self.mid_batch_embedded),
        self.fin_mid_sess_eb,
        [64, 32], 
        'relu',
        'pem_cur_att', 
        'din_mlp'
      )

      mask_pw_cur_router = self.pathway_router_simple(mid_sess_cur_eb_enhance, [32], 10,
                                                      'cur_router')  # b,10,1
      mid_sess_cur_eb_enhance = tf.reduce_sum((mid_sess_cur_eb_enhance * mask_pw_cur_router), axis=-2)  # b,eb
      out_fea_1 = mid_sess_cur_eb_enhance

      # 2. Pathway Matching Module
      mid_sess_cur_eb_pool = tf.reshape(mid_sess_cur_eb_enhance, [-1, 1, EMBEDDING_DIM])  # [B, 1, eb]
      mid_sess_his_eb_pool = mid_sess_his_eb_enhance  # [B, 18, eb]

      sess_score = tf.matmul(mid_sess_cur_eb_pool, tf.transpose(mid_sess_his_eb_pool, [0, 2, 1]))  # [B, 1, 18]
      attention_output_0 = din_attention(tf.squeeze(mid_sess_cur_eb_pool, 1), mid_sess_his_eb_pool,
                                         ATTENTION_SIZE, self.sess_mask, name_scope='attention_output_0')

      att_fea_0 = tf.reduce_sum(attention_output_0, 1)
      attention_output_1 = din_attention(self.item_eb, self.item_his_eb, ATTENTION_SIZE, self.mask,
                                         att_score=tf.squeeze(sess_score, 1), name_scope='attention_output1')
      att_fea_1 = tf.reduce_sum(attention_output_1, 1)

    inp = tf.concat(
      [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum,
      att_fea_0, att_fea_1, out_fea_0, out_fea_1], -1)

    # Fully connected layer
    logit = self.build_fcn_net(inp, use_dice=True)
    self.build_loss(logit)

  def pathway_router_simple(self, input_layer, hidden_units_list, output_dim, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
      net = input_layer  # b,18,10,eb or b,10,eb

      # net_reshape = tf.reduce_mean(net, axis=-1)  # b,18,10
      net_reshape = tf.squeeze(tf.concat(tf.split(net, net.shape[-2].value, axis=-2), axis=-1),
                               axis=-2)  # b,18,10*eb
      # second loop
      for i in range(len(hidden_units_list)):
        net_reshape = tf.layers.dense(inputs=net_reshape,
                                      units=hidden_units_list[i],
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.glorot_normal_initializer(),
                                      bias_initializer=tf.glorot_normal_initializer(),
                                      name='%s_fc_pw_%d' % (name, i))

      pw_router = tf.layers.dense(inputs=net_reshape,
                                  units=output_dim,
                                  activation=tf.nn.softmax,
                                  # activation=tf.nn.sigmoid,
                                  kernel_initializer=tf.glorot_normal_initializer(),
                                  bias_initializer=tf.glorot_normal_initializer(),
                                  name='%s_softmax_pw' % (name))  # b,18,10

      topk_vals, _ = tf.nn.top_k(pw_router, 5)  # B,18,5
      min_topk_vals = tf.reduce_min(topk_vals, axis=-1, keepdims=True)  # B,18,1

      pw_bool = tf.math.greater_equal(pw_router, min_topk_vals)  # B, 18, 10
      mask_val = tf.zeros_like(pw_router)
      mask_pw_router = tf.where(pw_bool, pw_router, mask_val)  # b,18,10

    return tf.expand_dims(mask_pw_router, axis=-1)  # b,18,10,1