 
import os
 

import sys
import time
import random
import datetime

import numpy as np
import tensorflow as tf

from model_all import *
from utils import *
from data_iterator import DataIterator

EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
best_auc = 0.0


def prepare_data(model_type, inputs, target, maxlen=100):
   
  lengths_x = [len(inp[4]) for inp in inputs]  
  lengths_sess = [len(inp[5]) for inp in inputs]  
  seqs_mid = [inp[3] for inp in inputs]  
  seqs_cat = [inp[4] for inp in inputs]  
  seqs_mid_sess = [inp[5] for inp in inputs]  
  seqs_cat_sess = [inp[6] for inp in inputs]  
  seqs_mid_tgt = [inp[7] for inp in inputs]  
  seqs_cat_tgt = [inp[8] for inp in inputs]  

  if maxlen is not None:
    new_seqs_mid = []
    new_seqs_cat = []
    new_lengths_x = []
    new_seqs_mid_sess = []
    new_seqs_cat_sess = []
    new_seqs_mid_tgt = []
    new_seqs_cat_tgt = []
    for l_x, inp in zip(lengths_x, inputs):
      if l_x > maxlen:
        new_seqs_mid.append(inp[3][l_x - maxlen:])
        new_seqs_cat.append(inp[4][l_x - maxlen:])
        new_lengths_x.append(maxlen)
      else:
        new_seqs_mid.append(inp[3])
        new_seqs_cat.append(inp[4])
        new_lengths_x.append(l_x)
     
    for l_sess, inp in zip(lengths_sess, inputs):
      if l_sess > 18:
        new_seqs_mid_sess.append(inp[5][l_sess - 18:])
        new_seqs_cat_sess.append(inp[6][l_sess - 18:])
        new_seqs_mid_tgt.append(inp[7][l_sess - 18:])
        new_seqs_cat_tgt.append(inp[8][l_sess - 18:])
      else:
        new_seqs_mid_sess.append(inp[5])
        new_seqs_cat_sess.append(inp[6])
        new_seqs_mid_tgt.append(inp[7])
        new_seqs_cat_tgt.append(inp[8])
    lengths_x = new_lengths_x
    seqs_mid = new_seqs_mid
    seqs_cat = new_seqs_cat
    seqs_mid_sess = new_seqs_mid_sess
    seqs_cat_sess = new_seqs_cat_sess
    seqs_mid_tgt = new_seqs_mid_tgt
    seqs_cat_tgt = new_seqs_cat_tgt
    if len(lengths_x) < 1:
      return None, None, None, None

  n_samples = len(seqs_mid)
  maxlen_x = np.max(lengths_x)
  maxlen_sess = 18

  mid_his = np.zeros((n_samples, maxlen_x)).astype('int64')
  cat_his = np.zeros((n_samples, maxlen_x)).astype('int64')
  mid_sess_his = np.zeros((n_samples, maxlen_sess, 10)).astype('int64')
  cat_sess_his = np.zeros((n_samples, maxlen_sess, 10)).astype('int64')
  mid_sess_tgt = np.zeros((n_samples, 18))
  cat_sess_tgt = np.zeros((n_samples, 18))

  fin_mid_sess = np.array([inp[9] for inp in inputs])  
  fin_cat_sess = np.array([inp[10] for inp in inputs])  

  mid_mask = np.zeros((n_samples, maxlen_x)).astype('float32')
  sess_mask = np.zeros((n_samples, maxlen_sess)).astype('float32')

  for idx, [s_x, s_y, sess_x, sess_y, sess_x_tgt, sess_y_tgt] in enumerate(
      zip(seqs_mid, seqs_cat, seqs_mid_sess, seqs_cat_sess, seqs_mid_tgt, seqs_cat_tgt)):
    mid_mask[idx, :lengths_x[idx]] = 1.
    sess_mask[idx, :lengths_sess[idx]] = 1.
    mid_his[idx, :lengths_x[idx]] = s_x
    cat_his[idx, :lengths_x[idx]] = s_y
     
    mid_sess_his[idx, :lengths_sess[idx]] = sess_x
    cat_sess_his[idx, :lengths_sess[idx]] = sess_y
    mid_sess_tgt[idx, :lengths_sess[idx]] = sess_x_tgt
    cat_sess_tgt[idx, :lengths_sess[idx]] = sess_y_tgt

  uids = np.array([inp[0] for inp in inputs])
  mids = np.array([inp[1] for inp in inputs])
  cats = np.array([inp[2] for inp in inputs])


  return uids, mids, cats, mid_his, cat_his, mid_mask, mid_sess_his, cat_sess_his, mid_sess_tgt, cat_sess_tgt, sess_mask, fin_mid_sess, fin_cat_sess, np.array(
    target), np.array(lengths_x)


def eval(sess, use_infonce, use_lal_infonce, model_type, test_data, model):
  loss_sum = 0.
  loss_infonce_sum = 0.
  loss_lal_infonce_sum = 0.
  accuracy_sum = 0.
  nums = 0
  stored_arr = []
  for idx, (src, tgt) in enumerate(test_data):
    nums += 1
    uids, mids, cats, mid_his, cat_his, mid_mask, mid_sess_his, cat_sess_his, mid_sess_tgt, cat_sess_tgt, sess_mask, fin_mid_sess, fin_cat_sess, target, sl = prepare_data(model_type, src, tgt)
    if use_infonce and use_lal_infonce:
      prob, loss, loss_infonce, loss_lal_infonce, acc = model.calculate(
        sess,
        [uids, mids, cats,
        mid_his, cat_his, mid_mask,
        mid_sess_his, cat_sess_his, mid_sess_tgt, cat_sess_tgt, sess_mask,
        fin_mid_sess, fin_cat_sess,
        target,
        sl]
      )
    elif use_infonce:
      prob, loss, loss_infonce, acc = model.calculate(
        sess,
        [uids, mids, cats,
        mid_his, cat_his, mid_mask,
        mid_sess_his, cat_sess_his, mid_sess_tgt, cat_sess_tgt, sess_mask,
        fin_mid_sess, fin_cat_sess,
        target,
        sl]
      )
    elif use_lal_infonce:
      prob, loss, loss_lal_infonce, acc = model.calculate(
        sess,
        [uids, mids, cats,
        mid_his, cat_his, mid_mask,
        mid_sess_his, cat_sess_his, mid_sess_tgt, cat_sess_tgt, sess_mask,
        fin_mid_sess, fin_cat_sess,
        target,
        sl]
      )
    else:
      prob, loss, acc = model.calculate(
        sess,
        [uids, mids, cats,
        mid_his, cat_his, mid_mask,
        mid_sess_his, cat_sess_his, mid_sess_tgt, cat_sess_tgt, sess_mask,
        fin_mid_sess, fin_cat_sess,
        target,
        sl]
      )
    loss_sum += loss
    if use_infonce:
      loss_infonce_sum += loss_infonce
    if use_lal_infonce:
      loss_lal_infonce_sum += loss_lal_infonce
    accuracy_sum += acc

    prob_1 = prob[:, 0].tolist()
    target_1 = target[:, 0].tolist()

    for p, t in zip(prob_1, target_1):
      stored_arr.append([p, t])

  test_auc = calc_auc(stored_arr)
  accuracy_sum = accuracy_sum / nums
  loss_sum = loss_sum / nums
  if use_infonce:
    loss_infonce_sum = loss_infonce_sum / nums
  if use_lal_infonce:
    loss_lal_infonce_sum = loss_lal_infonce_sum / nums

  global best_auc
  if best_auc < test_auc:
    best_auc = test_auc
     
  if use_infonce and use_lal_infonce:
    return test_auc, loss_sum, loss_infonce_sum, loss_lal_infonce_sum,  accuracy_sum
  elif use_infonce:
    return test_auc, loss_sum, loss_infonce_sum,  accuracy_sum
  elif use_lal_infonce:
    return test_auc, loss_sum, loss_lal_infonce_sum,  accuracy_sum
  else:
    return test_auc, loss_sum, accuracy_sum


def train(
    train_file="./taobao_2000w/local_train",
    test_file="./taobao_2000w/local_test",
    uid_voc="./taobao_2000w/uid_voc.pkl",
    mid_voc="./taobao_2000w/mid_voc.pkl",
    cat_voc="./taobao_2000w/cat_voc.pkl",
    batch_size=1024,
    maxlen=100,
    test_iter=None,
    print_iter=100,
    save_iter=100000,
    model_type='DNN',
    seed=2,
    tag="1st",
    use_infonce=0,
    ui_infonce_tau=0.07,
    aux_infonce_w=0.05,
    ui_proj_dim=16,
    use_lal_infonce=0,
    lal_infonce_tau=0.07,
    lal_infonce_w=0.01,
    infonce_loss_type="contrastive"
):
  start_ts = time.time()
  model_path = "./checkpoint/" + model_type + "_" + tag
  if not os.path.exists(model_path):
    os.mkdir(model_path)
  model_path = model_path + "/ckpt"

  gpu_options = tf.GPUOptions(allow_growth=True)
  gpu_config = tf.ConfigProto(gpu_options=gpu_options)
  with tf.Session(config=gpu_config) as sess:

    train_data = DataIterator(model_type, train_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen, shuffle_each_epoch=False)

    test_data = DataIterator(model_type, test_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen)

    n_uid, n_mid, n_cat = train_data.get_n()

    if model_type == 'DNN':
      model = Model_DNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, 
												use_infonce=use_infonce, ui_infonce_tau=ui_infonce_tau, 
												aux_infonce_w=aux_infonce_w, ui_proj_dim=ui_proj_dim,
                        use_lal_infonce=use_lal_infonce, lal_infonce_tau=lal_infonce_tau,
                        lal_infonce_w=lal_infonce_w, infonce_loss_type=infonce_loss_type, model_type=model_type)
    elif model_type == 'FM':
      model = Model_FM(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, 
                        use_infonce=use_infonce, ui_infonce_tau=ui_infonce_tau, 
                        aux_infonce_w=aux_infonce_w, ui_proj_dim=ui_proj_dim,
                        use_lal_infonce=use_lal_infonce, lal_infonce_tau=lal_infonce_tau,
                        lal_infonce_w=lal_infonce_w, infonce_loss_type=infonce_loss_type, model_type=model_type)
    elif model_type == 'DeepFM':
      model = Model_DeepFM(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, 
												use_infonce=use_infonce, ui_infonce_tau=ui_infonce_tau, 
												aux_infonce_w=aux_infonce_w, ui_proj_dim=ui_proj_dim,
                        use_lal_infonce=use_lal_infonce, lal_infonce_tau=lal_infonce_tau,
                        lal_infonce_w=lal_infonce_w, infonce_loss_type=infonce_loss_type, model_type=model_type)
    elif model_type == 'WideDeep':
      model = Model_WideDeep(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, 
												use_infonce=use_infonce, ui_infonce_tau=ui_infonce_tau, 
												aux_infonce_w=aux_infonce_w, ui_proj_dim=ui_proj_dim,
                        use_lal_infonce=use_lal_infonce, lal_infonce_tau=lal_infonce_tau,
                        lal_infonce_w=lal_infonce_w, infonce_loss_type=infonce_loss_type, model_type=model_type)
    elif model_type == 'DIN':
      model = Model_DIN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, 
												use_infonce=use_infonce, ui_infonce_tau=ui_infonce_tau, 
												aux_infonce_w=aux_infonce_w, ui_proj_dim=ui_proj_dim,
                        use_lal_infonce=use_lal_infonce, lal_infonce_tau=lal_infonce_tau,
                        lal_infonce_w=lal_infonce_w, infonce_loss_type=infonce_loss_type, model_type=model_type)
    elif model_type == 'DIEN':
      model = Model_DIEN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, 
												use_infonce=use_infonce, ui_infonce_tau=ui_infonce_tau, 
												aux_infonce_w=aux_infonce_w, ui_proj_dim=ui_proj_dim,
                        use_lal_infonce=use_lal_infonce, lal_infonce_tau=lal_infonce_tau,
                        lal_infonce_w=lal_infonce_w, infonce_loss_type=infonce_loss_type, model_type=model_type)
    elif model_type == 'CAN':
      model = Model_DIEN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_coaction=True, 
												use_infonce=use_infonce, ui_infonce_tau=ui_infonce_tau, 
												aux_infonce_w=aux_infonce_w, ui_proj_dim=ui_proj_dim,
                        use_lal_infonce=use_lal_infonce, lal_infonce_tau=lal_infonce_tau,
                        lal_infonce_w=lal_infonce_w, infonce_loss_type=infonce_loss_type, model_type=model_type)
    elif model_type == 'DBPMaN':
      model = Model_DBPMaN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, 
													use_infonce=use_infonce, ui_infonce_tau=ui_infonce_tau, 
													aux_infonce_w=aux_infonce_w, ui_proj_dim=ui_proj_dim,
                          use_lal_infonce=use_lal_infonce, lal_infonce_tau=lal_infonce_tau,
                          lal_infonce_w=lal_infonce_w, infonce_loss_type=infonce_loss_type, model_type=model_type)
    elif model_type == 'DeepMCP':
      model = Model_DeepMCP(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, 
													use_infonce=use_infonce, ui_infonce_tau=ui_infonce_tau, 
													aux_infonce_w=aux_infonce_w, ui_proj_dim=ui_proj_dim,
                          use_lal_infonce=use_lal_infonce, lal_infonce_tau=lal_infonce_tau,
                          lal_infonce_w=lal_infonce_w, infonce_loss_type=infonce_loss_type, model_type=model_type)
    elif model_type == 'DMR':
      model = Model_DMR(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, 
													use_infonce=use_infonce, ui_infonce_tau=ui_infonce_tau, 
													aux_infonce_w=aux_infonce_w, ui_proj_dim=ui_proj_dim,
                          use_lal_infonce=use_lal_infonce, lal_infonce_tau=lal_infonce_tau,
                          lal_infonce_w=lal_infonce_w, infonce_loss_type=infonce_loss_type, model_type=model_type)
    elif model_type == 'CL4CTR':
      model = Model_CL4CTR(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, 
													use_infonce=use_infonce, ui_infonce_tau=ui_infonce_tau, 
													aux_infonce_w=aux_infonce_w, ui_proj_dim=ui_proj_dim,
                          use_lal_infonce=use_lal_infonce, lal_infonce_tau=lal_infonce_tau,
                          lal_infonce_w=lal_infonce_w, infonce_loss_type=infonce_loss_type, model_type=model_type)     
    else:
      print("Invalid model_type : %s" % model_type)
      return
    print("model_type %s:" % model_type)

    begin_global_init_ts = time.time()
    sess.run(tf.global_variables_initializer())
    end_global_init_ts = time.time()
    print("Global init: %.4fs" % (end_global_init_ts - begin_global_init_ts))

    begin_local_init_ts = time.time()
    sess.run(tf.local_variables_initializer())
    end_local_init_ts = time.time()
    print("Local init: %.4fs" % (end_local_init_ts - begin_local_init_ts))

    iterations = 0
    lr = 0.001
    for itr in range(1):
      print("-------------------------")
      print("-------------------------")
      print("-------------------------")
      print("----------itr-------------")

      print(itr)
      
      loss_sum = 0.0
      loss_infonce_sum = 0.0
      loss_lal_infonce_sum = 0.0
      accuracy_sum = 0.

      for src, tgt in train_data:
        
        uids, mids, cats, mid_his, cat_his, mid_mask, mid_sess_his, cat_sess_his, mid_sess_tgt, cat_sess_tgt, sess_mask, fin_mid_sess, fin_cat_sess, target, sl = prepare_data(model_type, src, tgt, maxlen)
        if use_infonce and use_lal_infonce:
          loss, loss_infonce, loss_lal_infonce, acc = model.train(
            sess,
            [uids, mids, cats,
            mid_his, cat_his, mid_mask,
            mid_sess_his, cat_sess_his, mid_sess_tgt, cat_sess_tgt, sess_mask,
            fin_mid_sess, fin_cat_sess,
            target,
            sl,
            lr]
          )
          loss_infonce_sum += loss_infonce
          loss_lal_infonce_sum += loss_lal_infonce

        elif use_infonce:
          loss, loss_infonce, acc = model.train(
            sess,
            [uids, mids, cats,
            mid_his, cat_his, mid_mask,
            mid_sess_his, cat_sess_his, mid_sess_tgt, cat_sess_tgt, sess_mask,
            fin_mid_sess, fin_cat_sess,
            target,
            sl,
            lr]
          )
          loss_infonce_sum += loss_infonce

        elif use_lal_infonce:
          loss, loss_lal_infonce, acc = model.train(
            sess,
            [uids, mids, cats,
            mid_his, cat_his, mid_mask,
            mid_sess_his, cat_sess_his, mid_sess_tgt, cat_sess_tgt, sess_mask,
            fin_mid_sess, fin_cat_sess,
            target,
            sl,
            lr]
          )
          loss_lal_infonce_sum += loss_lal_infonce
        else:
          loss, acc = model.train(
            sess,
            [uids, mids, cats,
            mid_his, cat_his, mid_mask,
            mid_sess_his, cat_sess_his, mid_sess_tgt, cat_sess_tgt, sess_mask,
            fin_mid_sess, fin_cat_sess,
            target,
            sl,
            lr]
          )
        loss_sum += loss
        accuracy_sum += acc

        iterations += 1

        if iterations % print_iter == 0:
          if use_infonce and use_lal_infonce:
            tmp_loss = loss_sum / print_iter
            tmp_bce_loss = (loss_sum-aux_infonce_w*loss_infonce_sum-lal_infonce_w*loss_lal_infonce_sum)/print_iter
            tmp_infonce_loss = loss_infonce_sum/print_iter
            tmp_lal_infonce_loss = loss_lal_infonce_sum/print_iter
            tmp_acc =  accuracy_sum / print_iter
            print('%s: iterations: %d ----> train_loss: %.4f ---- train_bce_loss: %.4f ---- train_loss_infonce: %.4f ---- train_loss_lal_infonce: %.4f ---- train_accuracy: %.4f' % (datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S'), iterations, tmp_loss, tmp_bce_loss, tmp_infonce_loss, tmp_lal_infonce_loss, tmp_acc))
            loss_sum = 0.0
            loss_infonce_sum = 0.0
            loss_lal_infonce_sum = 0.0
            accuracy_sum = 0.0
          elif use_infonce:
            tmp_loss = loss_sum / print_iter
            tmp_bce_loss = (loss_sum-aux_infonce_w*loss_infonce_sum)/print_iter
            tmp_infonce_loss = loss_infonce_sum/print_iter
            tmp_acc =  accuracy_sum / print_iter
            print('%s: iterations: %d ----> train_loss: %.4f ---- train_bce_loss: %.4f ---- train_loss_infonce: %.4f ---- train_accuracy: %.4f' % (datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S'), iterations, tmp_loss, tmp_bce_loss, tmp_infonce_loss, tmp_acc))
            loss_sum = 0.0
            loss_infonce_sum = 0.0
            accuracy_sum = 0.0
          elif use_lal_infonce:
            tmp_loss = loss_sum / print_iter
            tmp_bce_loss = (loss_sum-lal_infonce_w*loss_lal_infonce_sum)/print_iter
            tmp_lal_infonce_loss = loss_lal_infonce_sum/print_iter
            tmp_acc =  accuracy_sum / print_iter
            print('%s: iterations: %d ----> train_loss: %.4f ---- train_bce_loss: %.4f ---- train_loss_lal_infonce: %.4f ---- train_accuracy: %.4f' % (datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S'), iterations, tmp_loss, tmp_bce_loss, tmp_lal_infonce_loss, tmp_acc))
            loss_sum = 0.0
            loss_lal_infonce_sum = 0.0
            accuracy_sum = 0.0
          else:
            print('%s: iterations: %d ----> train_loss: %.4f ---- train_accuracy: %.4f' % (datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S'), iterations, loss_sum / print_iter, accuracy_sum / print_iter))
            loss_sum = 0.0
            accuracy_sum = 0.0

        if (iterations % save_iter) == 0:
          print('save model iterations: %d' % iterations)
          model.save(sess, model_path + "--" + str(iterations))

      if use_infonce and use_lal_infonce:
        auc_, loss_, loss_infonce_, loss_lal_infonce_, acc_ = eval(sess, use_infonce, use_lal_infonce, model_type, test_data, model)
        bce_loss_ = loss_-aux_infonce_w*loss_infonce_-lal_infonce_w*loss_lal_infonce_
        print('Final: iterations: %d ----> final_auc: %.4f ---- final_loss: %.4f final_bce_loss: %.4f ---- final_infonce_loss: %.4f ---- final_lal_infonce_loss: %.4f ---- final_accuracy: %.4f' % (iterations, auc_, loss_, bce_loss_, loss_infonce_, loss_lal_infonce_, acc_))
        with open('./log_ablation/'+model_type+'--'+str(ui_infonce_tau)+'_'+str(aux_infonce_w)+'_'+str(ui_proj_dim)+'--'+str(lal_infonce_tau)+'_'+str(lal_infonce_w)+'.txt', mode='w') as f:
          f.write("auc:"+str(auc_)+"\n")
          f.write("bce_loss:"+str(bce_loss_)+"\n")
          f.write("infonce_loss:"+str(loss_infonce_)+"\n")
          f.write("lal_infonce_loss:"+str(loss_lal_infonce_)+"\n")
          f.write("acc:"+str(acc_)+"\n")
      elif use_infonce:
        auc_, loss_, loss_infonce_, acc_ = eval(sess, use_infonce, use_lal_infonce, model_type, test_data, model)
        bce_loss_ = loss_-aux_infonce_w*loss_infonce_
        print('Final: iterations: %d ----> final_auc: %.4f ---- final_loss: %.4f final_bce_loss: %.4f ---- final_infonce_loss: %.4f ---- final_accuracy: %.4f' % (iterations, auc_, loss_, bce_loss_, loss_infonce_, acc_))
        with open('./log_2000w_tal/'+model_type+'--'+str(ui_infonce_tau)+'_'+str(aux_infonce_w)+'_'+str(ui_proj_dim)+'.txt', mode='w') as f:
          f.write("auc:"+str(auc_)+"\n")
          f.write("bce_loss:"+str(bce_loss_)+"\n")
          f.write("infonce_loss:"+str(loss_infonce_)+"\n")
          f.write("acc:"+str(acc_)+"\n")
      elif use_lal_infonce:
        auc_, loss_, loss_lal_infonce_, acc_ = eval(sess, use_infonce, use_lal_infonce, model_type, test_data, model)
        bce_loss_ = loss_-lal_infonce_w*loss_lal_infonce_
        print('Final: iterations: %d ----> final_auc: %.4f ---- final_loss: %.4f final_bce_loss: %.4f ---- final_lal_infonce_loss: %.4f ---- final_accuracy: %.4f' % (iterations, auc_, loss_, bce_loss_, loss_lal_infonce_, acc_))
        with open('./cl4ctr_32_dp02/'+model_type+'--'+str(lal_infonce_tau)+'_'+str(lal_infonce_w)+'.txt', mode='w') as f:
          f.write("auc:"+str(auc_)+"\n")
          f.write("bce_loss:"+str(bce_loss_)+"\n")
          f.write("lal_infonce_loss:"+str(loss_lal_infonce_)+"\n")
          f.write("acc:"+str(acc_)+"\n")
      else:
        auc_, loss_, acc_ = eval(sess, use_infonce, use_lal_infonce, model_type, test_data, model)
        print('Final: iterations: %d ----> final_auc: %.4f ---- final_loss: %.4f ---- final_accuracy: %.4f' % (iterations, auc_, loss_, acc_))

      end_ts = time.time()
      print("Exp time: %.4fs" % (end_ts - start_ts))

if __name__ == '__main__':
  SEED = 3
  tf.set_random_seed(SEED)
  np.random.seed(SEED)
  random.seed(SEED)
  if sys.argv[1] == 'train':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[12])
    train(
    model_type=sys.argv[2], 
		seed=SEED, 
		tag=sys.argv[3], 
    use_infonce=int(sys.argv[4]),
    ui_infonce_tau=float(sys.argv[5]),
    aux_infonce_w=float(sys.argv[6]),
    ui_proj_dim=int(sys.argv[7]),
    use_lal_infonce=int(sys.argv[8]),
    lal_infonce_tau=float(sys.argv[9]),
    lal_infonce_w=float(sys.argv[10]),
    infonce_loss_type=str(sys.argv[11])
	)
  else:
    print('do nothing...')
