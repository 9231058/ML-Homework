# MLP-Homework
## Syllabus
- Learning Types
  - Supervised
    - Regression
    - Classification
  - Unsupervised
  - Reinforcement Learning
- Decision Tree
- Bayesian Classifier
- Logistic Regression
- Support Vector Machines
- Ensemble Methods
- Clustering
- Q-Learning
- Presentations

## Evaluation

- Midterm (20-25%)
- Final (35-40%)
- Homework (20-25%)
- Final Project (15-20%)

## Introduction
To have an environment for Machine Learning course:

```sh
python3 -m venv .
. ./bin/activate
python3 -m pip install -U numpy jupyter pandas matplotlib
python3 -m pip install -U scipy
python3 -m pip install -U mypy pycodestyle
```

In order to create PDF version of Jupyter Notebook you need the following packages:

```sh
sudo apt install pandoc
sudo apt install texlive-xetex
```

Then you can run the following command to create PDF:

```sh
jupyter-nbconvert notebook.ipynb --to pdf
```
