## Subscription Prediction
It is necessary to predict whether customers will subscribe to cars in the Auto Subscription service application.
The customer pays a fixed monthly payment and gets to use the car for a period of six months to three years.

## Installation
### Download the data
- Clone this repo to your computer.
- Get into the folder using `cd autosber-subscription`.
- Run `mkdir data`.
- Switch into the data directory using `cd data`.
- Download the data files from autosber-subscription into the `data` directory.
  - You can find the data [here](https://drive.google.com/file/d/1iW0GBTox3BMdn_kRiH88LIIj_OCp-3zI/view?usp=drive_link) and [here](https://drive.google.com/file/d/1YK_SOKFXhLaWdgdQglLxEoAsOMCA7M4x/view?usp=drive_link).
- Switch back into the `autosber-subscription` directory using `cd ..`.
## Install the requirements
- Install the requirements using `pip install -r requirements.txt`.
## Usage
- Run `mkdir processed` to create a directory for our processed datasets.
- Run `mkdir clean_data` to create a directory for our clean datasets.
- Run `python assemble.py` and after that `python data_preparation.py` to clean the datasets for train data generation.
  - This will create `data_sessions_clean.csv` and `data_hits_clean.csv` in the `clean_data` folder. 
- Run `python train_data_generation.py`.
  - This will create training data from `data_sessions_clean.csv` and `data_hits_clean.csv`.
  - It will add a file called `sberauto_train_data.csv` to the `clean_data` folder.
- Run `python pipeline.py`.
  - This will run cross validation across the training set, and print the accuracy score.
