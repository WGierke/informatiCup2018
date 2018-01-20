informatiCup2018 [![CircleCI](https://circleci.com/gh/WGierke/informatiCup2018.svg?style=svg&circle-token=00f4e65f31b3192e58b793d0282ba0af8c009b44)](https://circleci.com/gh/WGierke/informatiCup2018)
==============================

Predicting the optimal strategy for fueling for a given route ([task description](https://github.com/WGierke/informatiCup2018/blob/master/references/Intellitank.pdf)).

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │   └── build_features.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions
        │   ├── predict_model.py
        │   └── train_model.py
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py

--------
### Setup

Install all dependencies  
`pip install -r requirements.txt`  
Start the server  
`python3 src/serving/server.py`

### Credits
[Materialize](http://materializecss.com/)  
[bootstrap-material-datetimepicker](https://github.com/T00rk/bootstrap-material-datetimepicker)
