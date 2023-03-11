
import os
import numpy as np

# ----------
# faiss (Facebook AI Similarity Search)
# https://github.com/facebookresearch/faiss
# https://www.ariseanalytics.com/activities/report/20210304/
import faiss


##################################################################################################################
# ----------------------------------------------------------------------------------------------------------------
# 1-Flat
# 与えられたベクトルをメモリ空間にそのまま展開
# 検索は総当り。よって誤差は出ない。 それでもBLASに食わせ易く信じられないほど速い。
# メモリーにそのまま展開するためベクトルのメモリー内順序で処理しID管理はしない（できない）。
# 利用側がベクトルを追加した順番を覚えておかなければならないので取り回しが悪い。
# スクリプトなどで一時的なインデックスという使い方に良い。
# ----------------------------------------------------------------------------------------------------------------

# dimension
d = 64

# database size
nb = 100000

# nb of queries
nq = 10000


# ----------
np.random.seed(1234)
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.


# ----------
print(xb)
print(xq)


# ----------
# build the index
index = faiss.IndexFlatL2(d)
print(index.is_trained)

# add vectors to the index
index.add(xb)

# 100000 indices
print(index.ntotal)


# ----------s
# we want to see 4 nearest neighbors
k = 4

# # search from database
# D, I = index.search(xb[:5], k)
# print(I)
# print(D)

# actual search from database
# D: distance  I: indices
D, I = index.search(xq, k)
print(len(I))

# neighbors of the 5 first queries
print(I[:5])

# neighbors of the 5 last queries
print(I[-5:])



##################################################################################################################
# ----------------------------------------------------------------------------------------------------------------
# 2-IVFFlat
# IVF:
# 総当りに耐えられない数のベクトルを、クラスタリングして小さな nlist 個のインデックスに分けておく。
# 検索時は、指定したベクトルに近いクラスタ（インデックス）を nprobe 個を対象に検索する。
# クラスタリングの常として検索対象に入らなかったクラスタに、本当は近かったベクトルが入っている場合もあり、それらは検索結果から漏れる。
# ----------------------------------------------------------------------------------------------------------------

# IndexFlatL2
quantizer = faiss.IndexFlatL2(d)

# number of clusters
nlist = 100
k = 4
# here we specify METRIC_L2 by default, it performs inner - product search
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)


# ----------
index.train(xb)
assert index.is_trained

# default nprobe is 1
# nprobe: 探索する際にクエリに対して一番近い重心点を持つクラスタを見つけた後、
# その周りに存在するクラスタに対して、何個のクラスタを探索しに行くか、というパラメータ
print(index.nprobe)


# ----------
# add may be a bit slower as well
index.add(xb)

# actual search
D, I = index.search(xq, k)

# neighbors of the 5 last queries
print(I[-5:])


# ----------
# change nprobe
index.nprobe = 10

D, I = index.search(xq, k)
print(I[-5:])
