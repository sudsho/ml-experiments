# ml-experiments

Scratch notebooks I use to try new libraries and warm up on new topics. These are learning notes, not benchmarks. Nothing here is peer reviewed and nothing here should be treated as measured results.

Outputs are stripped from most notebooks. Some cells are stubs that just import a library or check a version; the goal was to try the API, not produce a result.

## quickstart

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
