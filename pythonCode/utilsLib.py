#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

from __future__ import absolute_import, division, print_function
import numbers
import numpy as np
import sklearn
import sklearn.metrics
import pickle
import pandas as pd
import math

def get_metric_func(metric='auc'):
  if metric == 'auc':
    return sklearn.metrics.roc_auc_score
  if metric == 'rmse':
    return lambda targ, pred: math.sqrt(sklearn.metrics.mean_squared_error(targ, pred))
  if metric == 'r2':
    return sklearn.metrics.r2_score
  if metric == 'mae':
    return sklearn.metrics.mean_absolute_error
  if metric == 'prc-auc': 
    return prc_auc

def prc_auc(targets, preds) -> float:
    precision, recall, _ = sklearn.metrics.precision_recall_curve(targets, preds)
    return sklearn.metrics.auc(recall, precision)

# unused in our version
# def calculateAUCs(t, p, metric, scaler=None):
#   print('calculateAUCs')
#   metric_func = get_metric_func(metric)
#   aucs = []
#   for i in range(p.shape[1]):
#     if metric == 'auc' or metric == 'prc-auc': # classification
#       targ = t[:, i] > 0.5
#       pred = p[:, i]
#       idx = np.abs(t[:, i]) > 0.5
#     else:
#       targ, pred = t, p
#     try:
#       aucs.append(metric_func(targ[idx], pred[idx]))
#     except ValueError:
#       aucs.append(np.nan)
#   return aucs

def calculateSparseAUCs(t, p, metric, scaler=None):
  metric_func = get_metric_func(metric)
  aucs = []
  # print(p.shape) # num tasks x num data
  for i in range(p.shape[0]):
    if metric == 'auc' or metric == 'prc-auc': # classification
      targ = t[i].data > 0.5
      pred = p[i].data
    else:
      targ, pred = t[i].data, p[i].data
      targ = scaler.inverse_transform_index(targ, i)
      pred = scaler.inverse_transform_index(pred, i)
    try:
      aucs.append(metric_func(targ, pred))
    except ValueError:
      aucs.append(np.nan)
  return aucs
  
def calculateAPs(t, p):
  aucs = []
  for i in range(p.shape[1]):
    aucs.append(0.5) # we aren't going to use the AP number anywhere so just continue to avoid any crashes
    continue
    targ = t[:, i] > 0.5
    pred = p[:, i]
    idx = np.abs(t[:, i]) > 0.5
    try:
      precision, recall, thresholds=sklearn.metrics.precision_recall_curve(targ[idx], pred[idx])
      area=sklearn.metrics.auc(recall, precision)
      aucs.append(area)
      #aucs.append(sklearn.metrics.average_precision_score(targ[idx], pred[idx]))
    except ValueError:
      aucs.append(np.nan)
  return aucs

def calculateSparseAPs(t, p):
  aucs = []
  for i in range(p.shape[0]):
    aucs.append(0.5) # we aren't going to use the AP number anywhere so just continue to avoid any crashes
    continue
    targ = t[i].data > 0.5
    pred = p[i].data
    try:
      precision, recall, thresholds=sklearn.metrics.precision_recall_curve(targ, pred)
      area=sklearn.metrics.auc(recall, precision)
      aucs.append(area)
      #aucs.append(sklearn.metrics.average_precision_score(targ, pred))
    except ValueError:
      aucs.append(np.nan)
  return aucs
  
def bestSettingsSimple(perfFiles, nrParams, takeMinibatch=[-1,-1,-1]):
  aucFold=[]
  for foldInd in range(0, len(perfFiles)):
    innerFold=-1
    aucParam=[]
    for paramNr in range(0, nrParams):
      #try:
      saveFile=open(perfFiles[foldInd][paramNr], "rb")
      aucRun=pickle.load(saveFile)
      saveFile.close()
      #except:
      #  pass
      if(len(aucRun)>0):
        if takeMinibatch[foldInd]<len(aucRun):
          aucParam.append(aucRun[takeMinibatch[foldInd]])
        else:
          aucParam.append(aucRun[-1])
    
    aucParam=np.array(aucParam)
    
    if(len(aucParam)>0):
      aucFold.append(aucParam)
  aucFold=np.array(aucFold)
  aucMean=np.nanmean(aucFold, axis=0)
  paramInd=np.nanmean(aucMean, axis=1).argmax()
  
  return (paramInd, aucMean, aucFold)
