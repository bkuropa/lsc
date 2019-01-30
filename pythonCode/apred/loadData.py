#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

import itertools
import numpy as np
import pandas as pd
import scipy
import scipy.io
import scipy.sparse
import pickle
from sklearn.feature_selection import VarianceThreshold

from typing import List

class StandardScaler:
  def __init__(self, means: np.ndarray = None, stds: np.ndarray = None, replace_nan_token=None):
    """Initialize StandardScaler, optionally with means and standard deviations precomputed."""
    self.means = means
    self.stds = stds
    self.replace_nan_token = replace_nan_token

  def fit(self, X: List[List[float]]) -> 'StandardScaler':
    """
    Learns means and standard deviations across the 0-th axis.

    :param X: A list of lists of floats.
    :return: The fitted StandardScaler.
    """
    X = X.todense()
    X = np.array(X).astype(float)
    X = np.where(X==0, np.ones(X.shape) * float('nan'), X)
    self.means = np.nanmean(X, axis=0)
    self.stds = np.nanstd(X, axis=0)
    self.means = np.where(np.isnan(self.means), np.zeros(self.means.shape), self.means)
    self.stds = np.where(np.isnan(self.stds), np.ones(self.stds.shape), self.stds)
    self.stds = np.where(self.stds == 0, np.ones(self.stds.shape), self.stds)

    return self

  def transform(self, X: List[List[float]]):
    """
    Transforms the data by subtracting the means and dividing by the standard deviations.

    :param X: A list of lists of floats.
    :return: The transformed data.
    """
    X = X.todense()
    X = np.array(X).astype(float)
    X = np.where(X==0, np.ones(X.shape) * float('nan'), X)
    transformed_with_nan = (X - self.means) / self.stds
    transformed_with_none = np.where(np.isnan(transformed_with_nan), 0, transformed_with_nan)

    return transformed_with_none

  def inverse_transform(self, X: List[List[float]]):
    """
    Performs the inverse transformation by multiplying by the standard deviations and adding the means.

    :param X: A list of lists of floats.
    :return: The inverse transformed data.
    """
    X = np.array(X).astype(float)
    X = np.where(X==0, np.ones(X.shape) * float('nan'), X)
    transformed_with_nan = X * self.stds + self.means
    transformed_with_none = np.where(np.isnan(transformed_with_nan), 0, transformed_with_nan)

    return transformed_with_none
  
  def inverse_transform_index(self, X: List[List[float]], i: int):
    """
    Performs the inverse transformation by multiplying by the standard deviations and adding the means.

    :param X: A list of lists of floats.
    :return: The inverse transformed data.
    """
    X = np.array(X).astype(float)
    X = np.where(X==0, np.ones(X.shape) * float('nan'), X)
    transformed_with_nan = X * self.stds[i] + self.means[i]
    transformed_with_none = np.where(np.isnan(transformed_with_nan), 0, transformed_with_nan)

    return transformed_with_none

f=open(dataPathFolds, "rb")
folds=pickle.load(f)
f.close()

f=open(dataPathSave + 'labelsHard.pckl', "rb")
targetMat=pickle.load(f)
sampleAnnInd=pickle.load(f)
targetAnnInd=pickle.load(f)
f.close()

targetMat=targetMat
if args.regression:
  originalTargetMat=targetMat
  targetScaler=StandardScaler()
  targetScaler.fit(targetMat)
  targetMat=scipy.sparse.csr_matrix(targetScaler.transform(targetMat))
targetMat=targetMat.copy().tocsr()
targetMat.sort_indices()
targetAnnInd=targetAnnInd
targetAnnInd=targetAnnInd-targetAnnInd.min()

folds=[np.intersect1d(fold, sampleAnnInd.index.values).tolist() for fold in folds]
targetMatTransposed=targetMat[sampleAnnInd[list(itertools.chain(*folds))]].T.tocsr()
targetMatTransposed.sort_indices()
trainPosOverall=np.array([np.sum(targetMatTransposed[x].data > 0.5) for x in range(targetMatTransposed.shape[0])])
trainNegOverall=np.array([np.sum(targetMatTransposed[x].data < -0.5) for x in range(targetMatTransposed.shape[0])])



#denseOutputData=targetMat.A
denseOutputData=None
sparseOutputData=targetMat



if datasetName=="static":
  f=open(dataPathSave+'static.pckl', "rb")
  staticMat=pickle.load(f)
  sampleStaticInd=pickle.load(f)
  # featureStaticInd=pickle.load(f)
  f.close()
  
  denseInputData=staticMat
  denseSampleIndex=sampleStaticInd
  sparseInputData=None
  sparseSampleIndex=None
  
  del staticMat
  del sampleStaticInd
elif datasetName=="semi":
  f=open(dataPathSave+'semi.pckl', "rb")
  semiMat=pickle.load(f)
  sampleSemiInd=pickle.load(f)
  # featureSemiInd=pickle.load(f)
  f.close()

  denseInputData=semiMat.A
  denseSampleIndex=sampleSemiInd
  sparseInputData=None
  sparseSampleIndex=None
  
  del semiMat
  del sampleSemiInd
elif datasetName=="ecfp":
  f=open(dataPathSave+'ecfp6.pckl', "rb")
  ecfpMat=pickle.load(f)
  sampleECFPInd=pickle.load(f)
  # featureECFPInd=pickle.load(f)
  f.close()

  denseInputData=None
  denseSampleIndex=None
  sparseInputData=ecfpMat
  sparseSampleIndex=sampleECFPInd
  sparseInputData.eliminate_zeros()
  sparseInputData=sparseInputData.tocsr()
  sparseInputData.sort_indices()
  
  del ecfpMat
  del sampleECFPInd
  
  sparsenesThr=0.0025
elif datasetName=="dfs":
  f=open(dataPathSave+'dfs8.pckl', "rb")
  dfsMat=pickle.load(f)
  sampleDFSInd=pickle.load(f)
  # featureDFSInd=pickle.load(f)
  f.close()

  denseInputData=None
  denseSampleIndex=None
  sparseInputData=dfsMat
  sparseSampleIndex=sampleDFSInd
  sparseInputData.eliminate_zeros()
  sparseInputData=sparseInputData.tocsr()
  sparseInputData.sort_indices()
  
  del dfsMat
  del sampleDFSInd
  
  sparsenesThr=0.02
elif datasetName=="ecfpTox":
  f=open(dataPathSave+'ecfp6.pckl', "rb")
  ecfpMat=pickle.load(f)
  sampleECFPInd=pickle.load(f)
  # featureECFPInd=pickle.load(f)
  f.close()
  
  f=open(dataPathSave+'tox.pckl', "rb")
  toxMat=pickle.load(f)
  sampleToxInd=pickle.load(f)
  # featureToxInd=pickle.load(f)
  f.close()

  denseInputData=None
  denseSampleIndex=None
  sparseInputData=scipy.sparse.hstack([ecfpMat, toxMat])
  sparseSampleIndex=sampleECFPInd
  sparseInputData.eliminate_zeros()
  sparseInputData=sparseInputData.tocsr()
  sparseInputData.sort_indices()
  
  del ecfpMat
  del sampleECFPInd
  del toxMat
  del sampleToxInd
  
  sparsenesThr=0.0025

gc.collect()



allSamples=np.array([], dtype=np.int64)
if not (denseInputData is None):
  allSamples=np.union1d(allSamples, denseSampleIndex.index.values)
if not (sparseInputData is None):
  allSamples=np.union1d(allSamples, sparseSampleIndex.index.values)
if not (denseInputData is None):
  allSamples=np.intersect1d(allSamples, denseSampleIndex.index.values)
if not (sparseInputData is None):
  allSamples=np.intersect1d(allSamples, sparseSampleIndex.index.values)
allSamples=allSamples.tolist()



if not (denseInputData is None):
  folds=[np.intersect1d(fold, denseSampleIndex.index.values).tolist() for fold in folds]
if not (sparseInputData is None):
  folds=[np.intersect1d(fold, sparseSampleIndex.index.values).tolist() for fold in folds]