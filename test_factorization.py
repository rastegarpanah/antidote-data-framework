import numpy as np
import scipy as sp
import pandas as pd
import timeit
import MF
reload(MF)

#Read Movielens Dataset
n_users=600
n_movies=2000
X, genres = MF.read_movielens(n_movies, n_users)

#Factorization parameters
rank = 4
lambda_ = 1.0 #Ridge regularizer parameter

#Initiate a recommender system of type ALS
RS = MF.als_MF(rank,lambda_)

#Initiate a recommender system of type lmafit
#RS = MF.lmafit_MF(rank)

start = timeit.default_timer()
pred,error = RS.fit_model(X)
time = timeit.default_timer() - start
print 'factorization time:',time

V = RS.get_V()
U = RS.get_U()
