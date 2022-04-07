import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

if __name__ == '__main__':
    from util import *
else:
    from data.util import *

class ContrastTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel are the images)
    Enhances contrast for each image
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):      
        return [apply_contrast(img) for img in X]


class ResizeTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel are the images)
    Resizes each image to a fixed resolution
    """
    def __init__(self, size=64):
        # transformer parameters
        self.size = size
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):      
        return np.array([reshape_(img, s=self.size) for img in X])


class colorhistTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel are the images)
    Resizes each image to a fixed resolution
    """
    def __init__(self,nbin=4):
        self.nbin=nbin
        pass
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        nbin=self.nbin
        # hist = np.zeros((nbin**3,len(X)))#X.shape[0]
        
        def local_hist(image,nbin):

            hist,y = np.histogram(np.floor(image/(256/nbin))[:,:,0]*nbin**2
                                  +np.floor(image/(256/nbin))[:,:,1]*nbin**1
                                  +np.floor(image/(256/nbin))[:,:,2],bins=np.arange(0,nbin**3+1),
                                  density=True)
            return hist
        
        return np.array([local_hist(img,nbin) for img in X])

# code snippit from https://kapernikov.com/tutorial-image-classification-with-scikit-learn/
class HogTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel are the images)
    Calculates hog features for each image using openCV HOG
    """
    def __init__(self, winSize=64, blockSize=64, blockStride=64, cellSize=8, nbins=9, resize=False):
        # HOG parameters
        self.winSize = winSize
        self.blockSize = blockSize
        self.blockStride = blockStride
        self.cellSize = cellSize
        self.nbins = nbins
        self.resize = resize
        
        self.derivAperture = 1
        self.winSigma = -1.
        self.histogramNormType = 0
        self.L2HysThreshold = 0.2
        self.gammaCorrection = 1
        self.nlevels = 64
        self.signedGradients = True
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if self.resize:
            # Resize the HOG parameters to have 
            size = X[0].shape[0]
            self.winSize = size
            self.blockSize = self.winSize
            self.blockStride = self.winSize
            self.cellSize = size//8

        hog = cv2.HOGDescriptor((self.winSize, self.winSize),
                            (self.blockSize, self.blockSize),
                            (self.blockStride, self.blockStride),
                            (self.cellSize, self.cellSize),
                            self.nbins, self.derivAperture, 
                            self.winSigma, self.histogramNormType, self.L2HysThreshold, 
                            self.gammaCorrection, self.nlevels, self.signedGradients)
        def local_hog(x):
            hog_features = hog.compute(x)
            return hog_features[:,0]
        
        return np.array([local_hog(img) for img in X])


class CorrMatrixTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel are the images)
    Calculates correlation matrix (using a method) of data and 
    cuts off correlated features if correlation is above treshold. 

    parameters:
    method: Method used in corr(), can be any of
        ['pearson', 'kendall', 'spearman']
    treshold: treshold value for correlation
    """
    def __init__(self, method='pearson', treshold=0.9):
        # Hyperparameters
        self.method = method
        self.treshold = treshold  
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=[]):
        # turn numpy X_train into pd dataframe and add column with labels
        row_names = [i for i in range(1,X.shape[0]+1)]
        column_names = ['HOG'+ str(i) for i in range(1,X.shape[1]+1)]
        pd_X = pd.DataFrame(data=X, index=row_names, columns=column_names)
        
        # Handles if y is given or not
        if  len(y) == 0:
            return X

        pd_X['class'] = y

        # Reduce features greedily using correlation matrix results
        corr = pd_X.corr(method=self.method)

        columns = np.full((corr.shape[0],), True, dtype=bool)        
        for i in range(corr.shape[0]-1):
            for j in range(i+1, corr.shape[0]):
                if abs(corr.iloc[i,j]) >= self.treshold:
                    if columns[j]:
                        columns[j] = False

        selected_columns = pd_X.columns[columns]
        X = pd_X[selected_columns]
        print(f"Using Correlation matrix: removed {np.count_nonzero(columns==False)} features.")
        return X

class AspectRatioTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel are the images)
    Enhances contrast for each image
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):      
        return [[get_ratio(img)] for img in X]
