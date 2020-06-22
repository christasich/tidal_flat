tidal_flat_0d
==============================

A zero-dimensional model of sediment accumulation on a tidal flat.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    |   |   ├── sutarkhali_pressure.csv     <- maniuplated raw export of pressure values from sutarkhali_data.mat.
    |   |   |                                  The original dataset had a sensor error from 09-Sep-2015 16:40:00 to 09-Sep-2015 17:50:00.
    |   |   |                                  This resulted in lower than expected tide heights. 4.19m 
    |   |   ├── sutarkhali_ssc.csv          <- raw export of suspended sediment cocentration values from sutarkhali_data.mat.
    |   |   ├──
    |   |   ├──
    |   |   └──
    │   ├── processed      <- The final, canonical data sets for modeling.
    |   |   ├── tides      <- Library of tides in feather format created using the OCE package in R
    |   |   ├── results    <- Sediment model results in feather format
    |   |   ├──
    |   |   ├──
    |   |   └──
    │   └── raw            <- The original, immutable data dump.
    |   |   ├── sutarkhali_data.mat     <- Initial tide dataset from Sutarkhali Forest Station provided by Rachel Bain
    |   |   ├──
    |   |   ├──
    |   |   ├──
    |   |   └──
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
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
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
