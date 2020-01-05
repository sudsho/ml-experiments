# ml-experiments

my scratch ML notebooks. textbook stuff, kaggle warmups, and now starting on deep learning.

## quickstart
```
pip install -r requirements.txt
jupyter notebook
```

## notebooks
### basics
- numpy-tricks.ipynb
- pandas-1.0-features.ipynb (jan 2020 features)
- pandas-basics.ipynb
- matplotlib-practice.ipynb
- seaborn-eda.ipynb

### classification
- iris-classifier-comparison.ipynb - LR/DT/RF/SVM
- titanic-feature-engineering.ipynb
- mnist-sklearn-digits.ipynb
- breast-cancer-diagnosis.ipynb
- decision-tree-vs-random-forest.ipynb
- gradient-boosting-intro.ipynb
- imbalanced-classification-smote.ipynb

### regression
- boston-housing-regression.ipynb

### preprocessing
- handling-missing-data.ipynb
- categorical-encoding.ipynb
- feature-scaling-comparison.ipynb

### model selection
- cross-validation-strategies.ipynb
- pipeline-and-gridsearch.ipynb

### unsupervised
- pca-and-dim-reduction.ipynb
- clustering-experiments.ipynb

## scripts
- utils.py - small data helpers
- plotting.py - mpl/seaborn helpers

## datasets
- mostly sklearn built-ins
- titanic from kaggle (file goes in `data/`, gitignored)
