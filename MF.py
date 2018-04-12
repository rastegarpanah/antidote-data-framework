import numpy as np
import als as als
import lmafit as lmafit
import pandas as pd
import numpy.ma as ma
from abc import ABCMeta, abstractmethod
reload(als)
reload(lmafit)

def read_movielens_small(n_movies, n_users, data_dir='MovieLens-small'):
    # get ratings
    df = pd.read_csv('Data/{}/ratings.csv'.format(data_dir))

    # create a dataframe with movie IDs on the rows
    # and user IDs on the columns
    ratings = df.pivot(index='movieId', columns='userId', values='rating')

    # put movie titles as index on rows
    movies = pd.read_csv('Data/{}/movies.csv'.format(data_dir))
    movieSeries = pd.Series(list(movies['title']),
                             index=movies['movieId'])
    ratings = ratings.rename(index=movieSeries)
    
    #read movie genres
    movie_genres = pd.Series(list(movies['genres']),index=movies['title'])
    movie_genres = movie_genres.apply(lambda s:s.split('|'))

    # select the top n_movies that have the most number of ratings
    num_ratings = (~ratings.isnull()).sum(axis=1)
    rows = num_ratings.nlargest(n_movies)
    ratings = ratings.loc[rows.index]
    
    # select the top n_users that have the most number of ratings
    num_ratings = (~ratings.isnull()).sum(axis=0)
    cols = num_ratings.nlargest(n_users)
    ratings = ratings[cols.index]

    # eliminate the users that have no ratings in this set
    null_columns = ratings.isnull().all(axis=0)
    null_column_ids = null_columns.index[null_columns]
    ratings = ratings.drop(null_column_ids, axis=1)
    ratings = ratings.T
    return ratings, movie_genres

def read_movielens_1M(n_movies, n_users, data_dir='MovieLens-1M'):
    # get ratings
    df = pd.read_table('Data/{}/ratings.dat'.format(data_dir),names=['UserID','MovieID','Rating','Timestamp'], 
                       sep='::', engine='python')

    # create a dataframe with movie IDs on the rows
    # and user IDs on the columns
    ratings = df.pivot(index='MovieID', columns='UserID', values='Rating')

    
    movies = pd.read_table('Data/{}/movies.dat'.format(data_dir),
                         names=['MovieID', 'Title', 'Genres'], 
                         sep='::', engine='python')
                         
    user_info = pd.read_table('Data/{}/users.dat'.format(data_dir),
                            names=['UserID','Gender','Age','Occupation','Zip-code'], 
                            sep='::', engine='python')
    user_info = user_info.rename(index=user_info['UserID'])[['Gender','Age','Occupation','Zip-code']]
    
    # put movie titles as index on rows
    movieSeries = pd.Series(list(movies['Title']), index=movies['MovieID'])
    ratings = ratings.rename(index=movieSeries)
    
    #read movie genres
    movie_genres = pd.Series(list(movies['Genres']),index=movies['Title'])
    movie_genres = movie_genres.apply(lambda s:s.split('|'))

    # select the top n_movies that have the most number of ratings
    num_ratings = (~ratings.isnull()).sum(axis=1)
    rows = num_ratings.nlargest(n_movies)
    ratings = ratings.loc[rows.index]
    
    # select the top n_users that have the most number of ratings
    num_ratings = (~ratings.isnull()).sum(axis=0)
    cols = num_ratings.nlargest(n_users)
    ratings = ratings[cols.index]

    ratings = ratings.T
    return ratings, movie_genres, user_info


class MF():
    
    __metaclass__ = ABCMeta
    
    def __init__(self, rank, lambda_=1e-6, ratings=None):
        self.rank = rank
        self.lambda_ = lambda_
        if ratings is not None:
            self.ratings = ratings
            self.num_of_known_ratings_per_user = (~self.ratings.isnull()).sum(axis=1)
            self.num_of_known_ratings_per_movie = (~self.ratings.isnull()).sum(axis=0)
    
    def set_ratings(self, ratings):
        self.ratings = ratings
        self.num_of_known_ratings_per_user = (~self.ratings.isnull()).sum(axis=1)
        self.num_of_known_ratings_per_movie = (~self.ratings.isnull()).sum(axis=0)
    
    def get_U(self):
        return pd.DataFrame(self.U, index = self.ratings.index)
    
    def get_V(self):
        return pd.DataFrame(self.V, columns = self.ratings.columns)
    
    @abstractmethod
    def fit_model():
        pass
    
        
class als_MF(MF):
    
    def fit_model(self, ratings=None, max_iter=100, threshold=0.001):
        X = self.ratings if ratings is None else ratings
        self.ratings = X
        self.U, self.V = als.als(X, self.rank, self.lambda_, max_iter, threshold)
        self.pred = pd.DataFrame(self.U.dot(self.V),
                                 index = X.index,
                                 columns = X.columns)
        self.error = ma.power(ma.masked_invalid(X-self.pred),2).sum()
        return self.pred, self.error


class lmafit_MF(MF):

    def fit_model(self, ratings=None, init=None):
        X = self.ratings if ratings is None else ratings
        self.ratings = X
        m, n = X.shape
        known_elements = np.where(~np.isnan(X.values))
        list_of_known_elements = zip(*known_elements)
        data = [X.values[coordinate] for coordinate in list_of_known_elements]
        self.U, self.V, opts = lmafit.lmafit_mc_adp(m, n, self.rank, known_elements, data, opts=init)        
        self.pred = pd.DataFrame(self.U.dot(self.V), index=X.index, columns=X.columns)
        self.error = ma.power(ma.masked_invalid(X-self.pred),2).sum()
        return self.pred, self.error