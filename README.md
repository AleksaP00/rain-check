# RainCheck

#### Description:

RainCheck is a command-line tool that predicts whether it will rain tomorrow. It downloads the daily weather history of a city from the free Open-Meteo API, trains a k-nearest-neighbours classifier (a scikit-learn pipeline), and uses today's forecast to predict tomorrow. It covers the whole life cycle of a small machine learning project: data, preparation, model, evaluation and use.

## How it works

The program has four commands. Each one leaves a file that the next one uses, so they never call each other:

```
fetch  ->  data/<city>.csv  ->  train  ->  models/model/  ->  predict
                                   |                |
                                   +---- evaluate --+
```

1. **fetch** finds the coordinates of a city (geocoding API), downloads the daily history (archive API) and saves it as CSV, together with a small JSON file holding the city and its coordinates.
2. **train** cleans the data, turns every day into an example (today's nine weather values as features, "did it rain tomorrow" as the label), splits the examples by time, fits the pipeline (scaler + k-NN) and saves a model folder: `model.pkl` (the pipeline), `config.json` (city, coordinates, features, `k`, metrics, dates, scikit-learn version) and `train.csv` / `test.csv` (the exact examples used).
3. **evaluate** loads the model folder and measures it on its saved test part.
4. **predict** downloads today's forecast for the city stored in the model and prints the chance of rain tomorrow.

## Installation

```
git clone https://github.com/AleksaP00/rain-check.git
cd rain-check
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```
python -m raincheck fetch --city "Novi Sad" --start 2018-01-01 --end 2026-10-01
python -m raincheck train --data data/novi_sad.csv --k 15
python -m raincheck evaluate
python -m raincheck predict
```

Add `-v` to any command to see debug messages. The end date of `fetch` should be at least a week in the past, because the reanalysis data arrives with a delay of about five days. Expected errors (unknown city, bad date, missing file, network problem) print a short message and exit with code 1, without a traceback.

Run the tests with `pytest -q`. They never use the network: the API is replaced with a fake response.

## Design choices

- **From scratch first, then scikit-learn.** Version `v1.0.0` (git tag) has the scaler and k-NN written by hand in pure Python. They were then replaced by scikit-learn's `StandardScaler` and `KNeighborsClassifier`.
- **Rain threshold of 1 mm.** A day counts as rainy if the precipitation sum is at least 1 mm. Reanalysis data often contains tiny values such as 0.1 mm, so a lower threshold would label almost every day as rainy.
- **Split by time.** The first 80% of the days are used for training and the last 20% for testing, without shuffling. Shuffling a time series lets the model see the future and makes the score look better than it is.
- **Scaler inside the pipeline.** `fit` learns the scaler's mean and standard deviation from the training part only, so information from the test part cannot leak into the model.
- **Scaling at all.** k-NN measures distances, so without scaling the sunshine duration (seconds, tens of thousands) would completely overpower the precipitation (millimetres).
- **Odd `k`.** With an odd number of neighbours the vote between two classes can never end in a tie.
- **A model folder, not just a model.** `config.json` records what the model was trained on, and `train.csv` / `test.csv` make `evaluate` repeatable even if the data file changes later.
- **No data in git.** `data/` and `models/` are created by the commands, so only the code that makes them is versioned.

## Limitations and next steps

- **Training-serving skew.** The model learns from archive (reanalysis) data but predicts from forecast data. The two sources differ, so the quality in use may be lower than on the test part.
- **One city per model** and a small set of nine daily features.
- **Pickle files.** `model.pkl` can run code when it is loaded, so load only model folders you made yourself. It may also not load in a different scikit-learn version; the version is saved in `config.json` and a warning is shown if it differs.
- Possible next steps: choosing `k` on a validation set, tuning the decision threshold, adding the day of the year as a seasonal feature, and running the tests in continuous integration.

## Files

- `raincheck/cli.py`: the command-line interface (`fetch`, `train`, `evaluate`, `predict`) and the results table.
- `raincheck/data_handler.py`: input checks, Open-Meteo downloads, CSV files, cleaning, examples and the time-based split.
- `raincheck/evaluation_metrics.py`: confusion counts, accuracy, precision, recall, F1 and the two baselines.
- `raincheck/model.py`: `build_pipeline`, the scikit-learn scaler + k-NN pipeline.
- `raincheck/artifacts.py`: saves and loads a model folder (`model.pkl`, `config.json`, `train.csv`, `test.csv`).
- `raincheck/__main__.py`: lets the package run as `python -m raincheck`.
- `tests/`: one test file per module; they never use the network.
- `pyproject.toml`: pytest settings (where the tests and the package are).
- `requirements.txt`: pinned versions of the dependencies.
