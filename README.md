# ml-experiments

my scratch ML notebooks. textbook stuff -> kaggle warmups -> deep learning -> diffusion + serving.

updates roughly weekly. nothing here is final.

still learning publicly. notebooks evolve as I figure things out.

**2020 pivot**: pytorch + tensorflow 2 + competitive kaggle.
**2021 pivot**: experiment tracking, lightning, transformers, more inference work.
**2022 pivot**: diffusion + accelerate + serving stacks (bentoml, triton, fastapi+redis).

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
- kaggle-house-prices-attempt.ipynb

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
started 2019 mostly sklearn. 2020 picked up pytorch + tf2, real kaggle attempts, started caring about hyperparam search and explainability. 2021 was experiment tracking, lightning, transformers, captum, quantization. 2022 added diffusion + a real serving stack. nothing here is polished, just working notes.

## causal + ab
- causal-inference-intro.ipynb
- bayesian-ab-test.ipynb

## quick wrap
going into 2021 with a much better feel for dl + tabular. the gap between sklearn me and torch me has narrowed.

## end of 2020 notes
favorite new tools this year: pytorch, optuna, lightgbm, shap.
took a while to get comfortable with pytorch but the explicit loop pays off when debugging.
going into 2021 i want to do a real recsys, more transformers stuff, and finally try fastapi.


## 2021 progress
newer 2021 notebooks (deeper dl + experiment tracking):
- captum-explainability.ipynb
- pytorch-quantization.ipynb
- onnx-export-and-runtime.ipynb
- ray-tune-hyperparameter.ipynb
- gradio-intro.ipynb
- dvc-data-versioning.ipynb
- pytorch-lightning-intro.ipynb
- transformer-from-scratch-tiny.ipynb
- vision-transformer-intro.ipynb
- huggingface-transformers-intro.ipynb
- huggingface-datasets-explore.ipynb
- bert-finetune-imdb.ipynb
- attention-from-scratch.ipynb
- mlflow-tracking-walkthrough.ipynb
- weights-and-biases-intro.ipynb
- albumentations-augmentations.ipynb
- timm-models-comparison.ipynb
- great-expectations-data-quality.ipynb
- pytest-fixtures-for-ml.ipynb
- cnn-from-scratch-numpy.ipynb

## 2022 progress
the year of inference + serving for me. lots of post-training engineering, less from-scratch training (already covered that in 2021).

### diffusion
- diffusion-models-intro-ddpm.ipynb (forward / reverse on 2d toy data, no unet)
- stable-diffusion-prompts.ipynb (api / inference, no training)

### llm + prompting (early)
- prompt-engineering-baseline.ipynb (zero-shot, few-shot, cot, role priming)
- in-context-learning-mini.ipynb
- whisper-transcription-quickstart.ipynb (sept 2022 release)

### serving / inference
- onnx-runtime-vs-pytorch.ipynb
- triton-inference-server-quickstart.ipynb
- bentoml-vs-fastapi-serve.ipynb
- fastapi-async-with-redis.ipynb
- streamlit-multipage-app.ipynb (1.10+ multipage)

### training infra
- pytorch-1.12-features.ipynb
- huggingface-accelerate-distributed.ipynb
- jax-vs-pytorch-microbench.ipynb

### ml ops + ci/cd
- mlflow-model-registry.ipynb (deeper than 2021 walkthrough)
- dvc-cml-cml-runners.ipynb (cml 0.2+, ci for ml)

### responsible ai + tabular
- fairlearn-disparate-impact.ipynb
- imbalanced-learn-2022.ipynb

### library previews
- pydantic-2-preview.ipynb (alpha at end of 2022)

## end of 2022 notes
favorite new tools this year: accelerate, diffusers, whisper, bentoml, triton.
biggest shift: less time on raw training, much more on inference, packaging, and prompts.
biggest surprise: how much prompt phrasing matters once you start actually measuring agreement on a small holdout.
going into 2023 i want to do a real rag pipeline, try a vector db, and finetune a small llm.


## 2023 progress
big pivot: most of the year is llms. rag pipelines, vector dbs, function calling, finetuning small models with lora.
chatgpt blew up around new year so by march most ml friends were already chasing prompts. trying not to skip the basics in the rush.

### llm + rag
- gpt-from-scratch-tiny-shakespeare.ipynb - karpathy walkthrough
- prompt-engineering-tricks.ipynb - cot, few-shot, role priming
- bert-vs-roberta-vs-deberta-comparison.ipynb

- langchain-quickstart.ipynb
- langchain-agents-with-tools.ipynb
- pinecone-vector-db-quickstart.ipynb
- chromadb-quickstart.ipynb
- llamaindex-vs-langchain.ipynb

### finetuning + quantization
- peft-lora-quickstart.ipynb
- bitsandbytes-8bit-loading.ipynb
- llama2-quantized-inference.ipynb (jul 2023 release)

### apis
- openai-function-calling.ipynb (jun 2023 release)
- anthropic-claude-api-quickstart.ipynb

### infra / serving for llms
- pytorch-2-compile-features.ipynb
- mlflow-llm-tracking.ipynb (mlflow 2.4+ llm features)
- pydantic-v2-migration.ipynb (jun 2023 release)

### sept-onward
- mistral-7b-quickstart.ipynb (sept 2023)
- gradio-vs-streamlit-llm-demos.ipynb
- diffusers-stable-diffusion-prompts.ipynb
- causal-dml-econml.ipynb (revisiting causal stuff with newer libs)

## end of 2023 notes
favorite new tools this year: langchain (with reservations), chroma, peft, bitsandbytes, anthropic api.
biggest shift: less from-scratch, much more wiring. half my notebooks this year are 'how do these libraries fit together'.
biggest surprise: how badly v0.0.x langchain breaks across minor versions. pinning religiously now.
going into 2024 i want a real agent loop, more eval rigor (evals, not vibes), and a real multimodal demo.

<!-- 2024 progress note 0 -->

<!-- 2024 progress note 1 -->

<!-- 2024 progress note 2 -->

<!-- 2024 progress note 3 -->
