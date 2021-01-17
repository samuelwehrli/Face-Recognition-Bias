import pandas as pd
import numpy as np
from scipy.linalg import orth
from scipy.spatial.distance import cosine


class H5Reader:
    pattern = r'C:\Daten\FaceRecognitionBias\vgg2\embeddings\{net}_pytorch_{race}_normed_BGR.h5'
    race_list = ['Caucasian','Indian','Asian','African']
    net_list = ['senet50_128','senet50_256','senet50_ft']
    
    
    def __init__(self, net):
        self.net = net
        
    @classmethod
    def net(cls, ind):
        return cls.net_list[ind]
        
    @classmethod
    def race(cls, ind):
        if type(ind) == str:
            return list(filter(lambda s: s.lower().startswith(ind.lower()),cls.race_list))[0]
        else:
            return cls.race_list[ind]
        
    def path(self, race):
        race = list(filter(lambda s: s.lower().startswith(race.lower()),self.race_list))[0]
        return self.pattern.format(net = self.net, race = race)
        
    def read(self, race_prefix, race_index = False, key = '/embeddings'):
        race = self.race(race_prefix)
        path = self.pattern.format(net = self.net, race = race)
        df = pd.read_hdf(path, key)
        if race_index:
            df['race_index'] = self.race_list.index(race)
        return df
    
    def read_all(self, N=None, select = 'head', key = '/embeddings'):
        out = []
        for race in self.race_list:
            df = self.read(race, race_index=True, key = key)
            if N is None:
                out.append(df)
            elif select == 'head':
                out.append(df.head(N))
            else:
                out.append(df.tail(N))    
                
        return pd.concat(out,0).set_index('race_index').reset_index()
    
    def read_pair_labels(self, N=None, select = 'head'):
        df = self.read_all(key='/images',N=N, select=select)
        return (df['class'] + '_' + df.img.apply(lambda s: s[:-4].split('.')[1])).values
        
                
    def keys(self, race_prefix):    
        race = self.race(race_prefix)
        path = self.pattern.format(net = self.net, race = race)
        with pd.HDFStore(path) as hdf:
            out = hdf.keys()
        return out
    
    @staticmethod
    def df2data(df):
        return df.filter(regex='\d').values, df.filter(regex='\D').values.flatten()
    
    @staticmethod
    def replacedata(df,X):
        return pd.concat([df.filter(regex='\D'), pd.DataFrame(X)],1)
    
# ----------------------------------------------------------------------------------------

# a class to project on subspaces
class ClusterCenterProjection:
    
    def fit(self,X, labels):
        centers = np.array([np.mean(X[labels==label,:],0) for label in np.unique(labels)])
        vectors = centers[1:,:] - centers[:1,:]
        self._basis = (vectors[0,:]/np.linalg.norm(vectors[0,:])).reshape([1,-1])
        # gram schmidt orthogonalization
        for iv in range(1,vectors.shape[0]):
            new_basis_vector = self.blind(vectors[iv,:])
            new_basis_vector = (new_basis_vector / np.linalg.norm(new_basis_vector)).reshape([1,-1])
            self._basis = np.concatenate([self._basis, new_basis_vector])
              

    def fit2(self,X, labels):
        centers = np.array([np.mean(X[labels==label,:],0) for label in np.unique(labels)])
        self._basis = orth((centers[1:,:] - centers[:1,:]).T).T
        
    def project(self,X, keepdims = False):
        out = np.dot(X,self._basis.T)
        if keepdims:
            out = np.dot(out, self._basis)
        return out
    
    def blind(self, X):
        return X - self.project(X,keepdims=True)
    
 # ------------------------------------------------------------------------------------------

class RFWDistances:
    pattern = r'C:\Daten\FaceRecognitionBias\RFW\{race}\{race}_pairs.txt'
    race_list = ['Caucasian','Indian','Asian','African']
    
    @staticmethod
    def _label1(s):
        s = s.split('\t')
        return s[0].split('.')[1] + '_' + s[1].zfill(4)

    @staticmethod
    def _label2(s):
        s = s.split('\t')
        if len(s) == 3:
            return s[0].split('.')[1] + '_' + s[2].zfill(4)
        else:
            return s[2].split('.')[1] + '_' + s[3].zfill(4)
        
    @staticmethod
    def _issame(s):
        return np.int8(len(s.split('\t'))==3)
        
    def __init__(self):
        df = [pd.read_csv(self.pattern.format(race=race),header=None).assign(race=race) for race in self.race_list]
        df = pd.concat(df,0).reset_index(drop=True)
        df['label1'] = df.race + '_' + df[0].apply(self._label1)
        df['label2'] = df.race + '_' + df[0].apply(self._label2)
        df['issame'] = df[0].apply(self._issame)
        self._df = df.drop(columns=[0])
        
    def _get_metric_empeddings(self, embeddings, labels):
        edf = pd.DataFrame(embeddings)
        edf.index = labels
        X1 = edf.reindex(self._df.label1).values
        X2 = edf.reindex(self._df.label2).values
        return X1, X2

    def get_distance_df(self, embeddings, labels):
        X1, X2 = self._get_metric_empeddings(embeddings, labels)
        df = self._df.copy()
        df['cos'] = np.array([cosine(X1[ii], X2[ii]) for ii in range(X1.shape[0])])
        df['eucl'] = np.linalg.norm(X1-X2,axis=1)
        return df
    
    def get_prod_df(self, embeddings, labels):
        X1, X2 = self._get_metric_empeddings(embeddings, labels)
        X1 = X1 / np.linalg.norm(X1, axis=1, keepdims=True)
        X2 = X2 / np.linalg.norm(X2, axis=1, keepdims=True)
        return pd.concat([self._df.copy(), pd.DataFrame(X1*X2)],1)
    
