#Copyright (C) 2018 Andreas Mayr
#Licensed under GNU General Public License v3.0 (see http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE)

if args.regression: 
  # lower hidden sizes for regression, as we observed this helps their stability and improves their performance on several regression datasets
  dictionary0 = {
    'basicArchitecture': ['selu', 'relu'],
    'l2Penalty': [0],
    'learningRate': [0.010, 0.1],
    'l1Penalty': [0.0],
    'idropout': [0.2, 0.0],
    'dropout': [0.5],
    'nrNodes': [100, 200, 300],
    'nrLayers': [3],
    'mom': [0.0]
  }

  dictionary1 = {
    'basicArchitecture': ['selu', 'relu'],
    'l2Penalty': [0],
    'learningRate': [0.010, 0.1],
    'l1Penalty': [0.0],
    'idropout': [0.2, 0.0],
    'dropout': [0.5],
    'nrNodes': [200],
    'nrLayers': [2,4],
    'mom': [0.0]
  }
else:
  dictionary0 = {
    'basicArchitecture': ['selu', 'relu'],
    'l2Penalty': [0],
    'learningRate': [0.010, 0.1],
    'l1Penalty': [0.0],
    'idropout': [0.2, 0.0],
    'dropout': [0.5],
    'nrNodes': [1024, 2048, 4096],
    'nrLayers': [3],
    'mom': [0.0]
  }

  dictionary1 = {
    'basicArchitecture': ['selu', 'relu'],
    'l2Penalty': [0],
    'learningRate': [0.010, 0.1],
    'l1Penalty': [0.0],
    'idropout': [0.2, 0.0],
    'dropout': [0.5],
    'nrNodes': [2048],
    'nrLayers': [2,4],
    'mom': [0.0]
  }

hyperParams0 = pd.DataFrame(list(itertools.product(*dictionary0.values())), columns=dictionary0.keys())
hyperParams1 = pd.DataFrame(list(itertools.product(*dictionary1.values())), columns=dictionary1.keys())
hyperParams=pd.concat([hyperParams0, hyperParams1], axis=0)
hyperParams.index=np.arange(len(hyperParams.index.values))