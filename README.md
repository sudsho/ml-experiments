# ml-experiments

my scratch ML notebooks. textbook stuff -> kaggle warmups -> deep learning.

updates roughly weekly. nothing here is final.

still learning publicly. notebooks evolve as I figure things out.

**2020 pivot**: starting with pytorch + tensorflow 2 + more competitive kaggle.

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
- streamlit_demo.py - tiny iris predictor

## datasets
- mostly sklearn built-ins
- titanic from kaggle (file goes in `data/`, gitignored)

## kaggle
- kaggle-titanic-leaderboard-attempt.ipynb

## deep learning (2020 pivot)
- pytorch-intro.ipynb
- pytorch-mnist-cnn.ipynb
- pytorch-cifar-resnet.ipynb
- tensorflow-2-quickstart.ipynb
- keras-vs-pytorch.ipynb

## halfway through 2020
main thing i learned so far: tuning matters more than model choice for the small kaggle stuff.
explicit pytorch loops are surprisingly readable once you write a few.

## nlp / embeddings
- imdb-sentiment-rnn.ipynb
- word2vec-from-scratch.ipynb
- glove-embeddings.ipynb

## skill arc
started 2019 mostly sklearn. 2020 was the year I picked up pytorch + tf2,
did real kaggle attempts, and started caring about hyperparameter search
and explainability. nothing here is polished, just my working notes.

## causal + ab
- causal-inference-intro.ipynb
- bayesian-ab-test.ipynb

## quick wrap
going into 2021 with a much better feel for dl + tabular. the gap between sklearn me and torch me has narrowed.

## end of 2020 notes
favorite new tools this year: pytorch, optuna, lightgbm, shap.
took a while to get comfortable with pytorch but the explicit loop pays off when debugging.
going into 2021 i want to do a real recsys, more transformers stuff, and finally try fastapi.
