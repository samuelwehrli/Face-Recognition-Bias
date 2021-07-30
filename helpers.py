import pandas as pd
import numpy as np
from scipy.linalg import orth
from scipy.spatial.distance import cosine

import sys, os
sys.path.insert(1,r'C:\Users\wehs\Documents\GitHub\Helpers')
import datasetloader as dsl


class H5Reader2:
    race_list = ['Caucasian','Indian','Asian','African']        
    gender_list = ['Female','Male']
    age_list = ['<30','30-45','45+']
    
    def __init__(self):
        # generate dataframe for the different datasets
        path = r'C:\Daten\FaceRecognitionBias\vgg2\embeddings\senet50_128*.h5'
        file_tagger = lambda file: {'race': os.path.split(file)[1].split('_')[3], 'model':'VGG128'} 
        ddf = dsl.h5_files_ddf(path, file_tagger=file_tagger) 
       
        path = r'C:\Daten\FaceRecognitionBias\vgg2\embeddings\senet50_256*.h5'
        file_tagger = lambda file: {'race': os.path.split(file)[1].split('_')[3], 'model':'VGG256'} 
        ddf = ddf.append(dsl.h5_files_ddf(path, file_tagger=file_tagger)) 
       
        path = r'C:\Daten\FaceRecognitionBias\vgg2\embeddings\senet50_ft*.h5'
        file_tagger = lambda file: {'race': os.path.split(file)[1].split('_')[3], 'model':'VGGft'} 
        ddf = ddf.append(dsl.h5_files_ddf(path, file_tagger=file_tagger)) 
       
        path = r'C:\Daten\FaceRecognitionBias\OpenFace\embeddings\*.h5'
        file_tagger = lambda file: {'race': os.path.split(file)[1].split('_')[1], 'model':'OpenFace'}        
        ddf = ddf.append(dsl.h5_files_ddf(path, file_tagger=file_tagger))
    
        path = r'C:\Daten\FaceRecognitionBias\FRbias-facenet\results\embeddings\*.h5'
        file_tagger = lambda file: {'race': os.path.split(file)[1].split('.')[0].split('_')[1], 'model':'FaceNet'}        
        ddf = ddf.append(dsl.h5_files_ddf(path, file_tagger=file_tagger, ))
    
        self.model_list = list(ddf['tag_model'].unique())
        ddf['tag_race'] = ddf['tag_race'].astype('category').cat.set_categories(self.race_list)     
        ddf['tag_model'] = ddf['tag_model'].astype('category').cat.set_categories(self.model_list)     
        self.ddf = ddf.sort_values(by=['tag_model','tag_race']).reset_index(drop=True)
        
        # generate dataframe with gender and age for each subject
        path = r'C:\Daten\FaceRecognitionBias\age-gender-estimation\results\*.h5'
        file_tagger = lambda file: {'race': os.path.split(file)[1].split('_')[1]}        
        agddf = dsl.h5_files_ddf(path, file_tagger=file_tagger)
        agdf, _ = dsl.load(agddf.query("arg_key=='images'"))
        agdf['subject'] = agdf['class'] + '_' +agdf.subject.apply(lambda s: s.split('.')[1]).values     
        agdf = agdf.groupby('subject')[['age','gender']].mean()

        gfun = lambda x: self.gender_list[x < 0.5]
        agdf['gender'] = (agdf['gender'].apply(gfun)
                          .astype('category')
                          .cat.set_categories(self.gender_list)  )

        afun = lambda x: self.age_list[1-(x<30)+(x>=45)] 
        agdf['age'] = (agdf['age'].apply(afun)
                       .astype('category')
                       .cat.set_categories(self.age_list) )  
        self.agegen_df = agdf        
                
    def read(self, model,embeddings_key='embeddings'):
        """
        Reads the data for a given model (VGG128, VGG256, VGGft, OpenFace)
        
        Paremeters
        ----------
        
        model: string
            name of the model
            
        Output
        ------
        
        X: The embeddings
        
        df: Dataframe with different labels
            They can be mapped to numberical values by df.race.cat.codes.values
        """
        # load empeddings
        sel = (self.ddf['tag_model'] == model) & (self.ddf['arg_key']==embeddings_key) 
        Xdf, tag_df = dsl.load(self.ddf[sel])
        tag_df['race'].cat.set_categories(self.race_list,inplace=True) 
        
        # load image and subject labels
        sel = (self.ddf['tag_model'] == model) & (self.ddf['arg_key']=='images') 
        df, _ = dsl.load(self.ddf[sel])      
        df['img'] = df['class'] + '_' + df.img.apply(lambda s: s.split('.')[1]).values
        df['subject'] = df['class'] + '_' + df.subject.apply(lambda s: s.split('.')[1]).values
        
        # join
        df = pd.concat([tag_df,df],1)
        df = df.drop(columns=['class','model'])
       
        # add gender and age
        for col in ['age','gender']:
            df.insert(2,col,self.agegen_df[col].reindex(df.subject).values)
        
        return Xdf.values, df

# ----------------------------------------------------------------------------------------
    
class H5Reader:
    pattern = r'C:\Daten\FaceRecognitionBias\vgg2\embeddings\{net}_pytorch_{race}_normed_BGR.h5'
    race_list = ['Caucasian','Indian','Asian','African']
    net_list = ['senet50_128','senet50_256','senet50_ft']
    
    age_gender_pattern = r'C:\Daten\FaceRecognitionBias\age-gender-estimation\results\weights.28-3.73_{race}.h5'

    
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
    
    def read_img_labels(self, N=None, select = 'head', idonly = False):
        df = self.read_all(key='/images',N=N, select=select)
        out = (df['class'] + '_' + df.img.apply(lambda s: s[:-4].split('.')[1])).values
        if idonly:
            out = [label.split('_')[1] for label in out]
        return out
        
    read_pair_labels = read_img_labels  # old notation   

    def read_age_gender(self, N=None, select = 'head', key = 'images'):
        agdf = []
        for race in self.race_list:
            path = self.age_gender_pattern.format(race = race)
            agdf.append(pd.read_hdf(path, key))
        agdf = pd.concat(agdf,0).reset_index(drop=True)
        agdf['id'] = agdf.subject.apply(lambda s: s.split('.')[1])
        agdf = agdf.groupby('id')[['age','gender']].mean()
        tmp = agdf.gender.values
        tmp[tmp==0.5] = np.nan
        agdf.gender = tmp.round().astype(int)
        img_ids = self.read_img_labels(N=N, select=select, idonly=True)
        agdf = agdf.reindex(img_ids).reset_index()
        return agdf
        
                
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
    
    
# function for blinding
def blind(Xin, *args, verbose = True):
    X = Xin.astype(np.float64)
    Ne = X.shape[1]  # size of the empedding space
    V = np.zeros([0,Ne])
    
    V = []
    for y in args:
        y_unique = np.unique(y[np.isfinite(y)]).astype(int)
        centers = np.array([np.mean(X[y==k,:],0) for k in y_unique])  # centers of the clusters
        V.append(np.array([centers[k] - np.mean(centers[y_unique!=k,:],0) for k in y_unique])) 
        
    V = np.concatenate(V,0)
    U = V / np.linalg.norm(V,axis=1,keepdims=True) # normalizing
    B = orth(U.T).T  # orthogal basis spanned by the vectors
    P = np.eye(Ne) - np.matmul(B.T,B)  # construct projection matrix
    Xb = np.matmul(X,P) # projection
    
    if verbose:
        print('eigenvalues of B =',np.linalg.svd(B)[1]) # as a check

    return Xb.astype(np.float32), U.astype(np.float32)

# ------------------------------------------------------------------------------------------

from sklearn.model_selection import GroupKFold
class ModelBenchmarker:
    def __init__(self, imglabel, n_splits=2):
        group_kfold = GroupKFold(n_splits=n_splits)
        self.indices = [res for res in group_kfold.split(imglabel, groups=imglabel)]
                  
    def predict(self, clf, X, y):
        y_pred = np.zeros(y.shape)
        for itrain,itest in self.indices:            
            X_train = X[itrain,:] 
            y_train = y[itrain] 
            good = np.isfinite(y_train)
            X_train = X_train[good,:]
            y_train = y_train[good]
   
            clf.fit(X_train, y_train)
            y_pred[itest] = clf.predict(X[itest,:])
        return y_pred.astype(np.uint8)
        
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
    
# ------------------------------------------------------------------------------------------

# add column to hdf store
def hdf_add_col(path, key, col, data):
    with pd.HDFStore(path) as store:
        keys = store.keys()
    if key in [key.strip('/') for key in keys]:
        df = pd.read_hdf(path, key)
        df[col] = data
    else:
        df = pd.DataFrame(data,columns=[col])
    df.to_hdf(path, key)
    
    
# generator function to loop over all embeddings
def X_generator(X_dict, y_args_dict,verbose=False,cosine=True):
    for model, X in X_dict.items():
        for blinding, y_args in y_args_dict.items():
            if len(y_args) == 0:
                yield model+'-'+blinding+'-e',X
            else:
                Xb = blind(X,*y_args,verbose=verbose)[0]
                yield model+'-'+blinding+'-e',Xb
                if cosine:
                    yield model+'-'+blinding+'-c',Xb/np.linalg.norm(Xb,axis=1,keepdims=True)
    
