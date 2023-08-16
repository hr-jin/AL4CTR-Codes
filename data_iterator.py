import shuffle
import cPickle as pkl

import numpy as np

def load_dict(filename):
  with open(filename, 'rb') as f:
    return pkl.load(f)

class DataIterator:

  def __init__(
      self,
      model_type,
      source,
      uid_voc,
      mid_voc,
      cat_voc,
      batch_size=128,
      maxlen=100,
      skip_empty=False,
      shuffle_each_epoch=False,
      sort_by_length=True,
      max_batch_size=20,
      minlen=50
  ):
    self.model_type = model_type
    if shuffle_each_epoch:
      self.source_orig = source
      self.source = shuffle.main(self.source_orig, temporary=True)
    else:
      self.source = open(source, 'r')
    self.source_dicts = []

    for source_dict in [uid_voc, mid_voc, cat_voc]:
      self.source_dicts.append(load_dict(source_dict))

    self.batch_size = batch_size

    self.maxlen = maxlen
    self.minlen = minlen

    self.skip_empty = skip_empty

    self.n_uid = len(self.source_dicts[0])
    self.n_mid = len(self.source_dicts[1])
    self.n_cat = len(self.source_dicts[2])

    self.shuffle = shuffle_each_epoch
    self.sort_by_length = sort_by_length

    self.source_buffer = []
    self.k = batch_size * max_batch_size

    self.end_of_data = False

    print("self.batch_size: %d" % self.batch_size)
    print("self.maxlen:  %d" % self.maxlen)
    print("self.minlen:  %d" % self.minlen)
    print("self.skip_empty:  %d" % self.skip_empty)
    print("self.n_uid:  %d" % self.n_uid)
    print("self.n_mid:  %d" % self.n_mid)
    print("self.n_cat:  %d" % self.n_cat)
    print("self.shuffle:  %d" % self.shuffle)
    print("self.sort_by_length:  %d" % self.sort_by_length)
    print("self.k:  %d" % self.k)
    print("self.end_of_data:  %d" % self.end_of_data)

  def get_n(self):
    return self.n_uid, self.n_mid, self.n_cat

  def __iter__(self):
    return self

  def reset(self):
    if self.shuffle:
      self.source = shuffle.main(self.source_orig, temporary=True)
    else:
      print("reset: seek-0")
      self.source.seek(0)

  def next(self):
    if self.end_of_data:
      self.end_of_data = False
      self.reset()
      print("Stop begin")
      raise StopIteration

    source = []
    target = []

    if len(self.source_buffer) == 0:
      for _ in range(self.k):
        ss = self.source.readline()
        if ss == "":
          break
        self.source_buffer.append(ss.strip("\n").split("\t"))

      # sort by  history behavior length
      if self.sort_by_length:
        his_length = np.array([len(s[4].split(",")) for s in self.source_buffer])
        tidx = his_length.argsort()

        _sbuf = [self.source_buffer[i] for i in tidx]
        self.source_buffer = _sbuf
      else:
        self.source_buffer.reverse()

    if len(self.source_buffer) == 0:
      self.end_of_data = False
      self.reset()
      print("Stop Middle")
      raise StopIteration

    try:

      # actual work here
      while True:

        # read from source file and map to word index
        try:
          ss = self.source_buffer.pop()
        except IndexError:
          break

        uid = self.source_dicts[0][ss[1]] if ss[1] in self.source_dicts[0] else 0
        mid = self.source_dicts[1][ss[2]] if ss[2] in self.source_dicts[1] else 0
        cat = self.source_dicts[2][ss[3]] if ss[3] in self.source_dicts[2] else 0

        mid_list = [self.source_dicts[1][mmid] for mmid in ss[4].split(',')]
        cat_list = [self.source_dicts[2][ccat] for ccat in ss[5].split(',')]

        # read from source file and map to word index
        if self.minlen != None:
          if len(mid_list) <= self.minlen:
            continue
        if self.skip_empty and (not mid_list):
          continue

        fin_mid_sess = mid_list[-10:] # Last 10 items
        fin_cat_sess = cat_list[-10:] # Last 10 categorys

        mid_sess_list = []
        cat_sess_list = []
        mid_sess_tgt = []
        cat_sess_tgt = []

        # if self.model_type == "DBPMaN":
        idx = len(mid_list)-5
        while idx >= 10:
          mid_sess_list.insert(0, mid_list[idx-10: idx])
          mid_sess_tgt.insert(0, mid_list[idx])
          cat_sess_list.insert(0, cat_list[idx-10: idx])
          cat_sess_tgt.insert(0, cat_list[idx])
          idx -= 5

        source.append([uid, mid, cat, mid_list, cat_list, mid_sess_list, cat_sess_list, mid_sess_tgt, cat_sess_tgt, fin_mid_sess, fin_cat_sess])

        target.append([float(ss[0])])

        if len(source) >= self.batch_size or len(target) >= self.batch_size:
          break

    except IOError:
      print("Stop End")
      self.end_of_data = True

    # all sentence pairs in maxibatch filtered out because of length
    if len(source) == 0 or len(target) == 0:
      source, target = self.next()

    return source, target

if __name__ == '__main__':
  test_iter = DataIterator(
    source="local_debug",
    uid_voc="uid_voc.pkl",
    mid_voc="mid_voc.pkl",
    cat_voc="cat_voc.pkl",
    batch_size=333,
    maxlen=100
  )
  for src, tar in test_iter:
    print(len(src))