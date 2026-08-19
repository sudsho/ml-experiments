# ml-experiments

Scratch notebooks I use to try new libraries and warm up on new topics. These are learning notes, not benchmarks. Nothing here is peer reviewed and nothing here should be treated as measured results.

Outputs are stripped from most notebooks. Some cells are stubs that just import a library or check a version; the goal was to try the API, not produce a result.

## Quick start (runs offline)

A small offline smoke runs two representative experiments end to end on CPU
with no downloads and no network. Both use sklearn built-in datasets, so there
is nothing to fetch. Needs numpy, scikit-learn, and pytest only.

```
make smoke     # or: python scripts/smoke.py
make test      # or: python -m pytest tests/ -v
```

Real output from `python scripts/smoke.py`:

```
============================================================
ml-experiments offline smoke
============================================================

[A] iris-classifier-comparison (sklearn load_iris)
    samples: 150
    held out accuracy:
      LR   0.9333
      DT   0.9333
      RF   0.9111
      SVM  0.9778
    5 fold cross validation mean:
      LR   0.9733
      DT   0.9533
      RF   0.9667
      SVM  0.9800

[B] breast-cancer-diagnosis (sklearn load_breast_cancer)
    samples: 569
    scaled LR held out accuracy: 0.9580
    scaled LR ROC AUC:           0.9952
    xgboost 5 fold CV mean:      0.9666
    classification report:
                    precision    recall  f1-score   support

         malignant       0.94      0.94      0.94        53
            benign       0.97      0.97      0.97        90

          accuracy                           0.96       143
         macro avg       0.96      0.96      0.96       143
      weighted avg       0.96      0.96      0.96       143

============================================================
SMOKE OK - both experiments trained and cleared metric floors
============================================================
```

And `python -m pytest tests/ -v`:

```
tests/test_smoke.py::test_iris_experiment_produces_metrics PASSED         [ 50%]
tests/test_smoke.py::test_breast_cancer_experiment_produces_metrics PASSED [100%]
2 passed
```

### what the smoke covers, and what it does not

The smoke covers the two sklearn classical-ML experiments end to end
(`scripts/smoke.py` extracts the core of `iris-classifier-comparison.ipynb` and
`breast-cancer-diagnosis.ipynb`). It asserts each experiment trained and cleared
a metric floor (RF and every CV mean above 0.85 on iris, scaled LR above 0.90
accuracy and 0.95 ROC AUC on breast cancer). The xgboost comparison runs only if
xgboost is already importable; the smoke never installs it and skips it cleanly
if it is missing. It does not touch the deep-learning, LLM-API, or infra
notebooks, which need heavy pins, GPUs, model downloads, or API keys and are not
part of the offline path.

## notebook quickstart

```
pip install -r requirements.txt
jupyter notebook
```

Only install the pins for topics you actually want to open. The full requirements set is heavy.

## what is in here

Rough groups. File names should be self explanatory.

- classical ML: iris, titanic, breast cancer, boston housing, gradient boosting, decision trees vs random forest, xgboost, lightgbm, imbalanced SMOTE.
- preprocessing: missing data, categorical encoding, feature scaling.
- model selection: cross validation, pipeline plus gridsearch, optuna vs gridsearch.
- unsupervised: PCA, clustering.
- data / plotting basics: numpy, pandas, matplotlib, seaborn.
- deep learning: pytorch intro, mnist CNN, cifar resnet, tensorflow 2 quickstart, keras vs pytorch, lightning intro, jax vs pytorch microbench.
- NLP: word2vec from scratch, glove, imdb sentiment RNN, huggingface transformers intro, bert finetune, attention from scratch, transformer from scratch (tiny), vision transformer intro.
- serving / infra tries: onnx export, triton quickstart, bentoml vs fastapi, fastapi with redis, streamlit multipage.
- experiment tracking / MLOps tries: mlflow, wandb, tensorboard, dvc, great expectations, ray tune, captum, pytest fixtures.
- causal / AB: causal-inference-intro, bayesian AB test, causal-dml-econml.
- LLM API pokes: openai function calling, anthropic api quickstart, gemini api quickstart, langchain quickstart, langchain agents, chromadb quickstart, pinecone quickstart, llamaindex vs langchain, gguf/llamacpp, bitsandbytes 8 bit, peft LoRA, whisper.
- pytorch release notes walk throughs: pytorch-1.12, pytorch-2 compile, pytorch-2.3, pytorch-2.5.
- pydantic / uv / ruff: version bumps as they landed.

## scripts

- utils.py, plotting.py: small helpers.
- streamlit_demo.py: tiny iris predictor.
- tf_serving_test.py: manual smoke script for a local TF serving container.

## datasets

Mostly sklearn built ins. Titanic file from kaggle goes in `data/` (gitignored).

## caveats

- No benchmarks. Any numbers inside notebooks are illustrative or scratch, not measured throughput or accuracy on a real run.
- No CI, no test suite, no run artifacts checked in.
- Duplicate markdown notes may appear across notebooks. This is a scratch pile, not a curated knowledge base.
- Library pins in requirements.txt track what I had installed most recently. Older notebooks were written against older versions and may not re run cleanly on the current pins.
