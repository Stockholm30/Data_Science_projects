## Credit Risk Prediction
This work is devoted to predicting the probability of default of customers who take out a loan. The metric value on the test dataset must be at least 0.75 according to ROC-AUC. For banks, this task is very important, since it depends on whether the bank goes bankrupt or not.
## Installation 
### Download the data
- Clone this repo to your computer.
- Get into the folder using `cd Credit_risk`.
- Run `mkdir processed`.
- Run `mkdir target_data`.
- Switch into the `target_data` directory using `cd target_data`.
- Download the data files with targets [here](https://drive.google.com/file/d/1KNnfCT7OueH1gAYF68Bx03H6dLiI0pmu/view?usp=drive_link).
- Run `../` to return in `Credit_risk` folder.
- Download the data files with client information.
  - You can find the data [here](https://drive.google.com/drive/folders/14npslKbipCFP5A9b-Tf46TUys6WaQgGY?usp=drive_link).
## Usage
- Run `python uploading_script.py` to create dataset for learning.
  - Data preprocessing is explained in notebooks : cleaning-credit_data.ipynb, EDA_credit_data.ipynb, modeling_default_prediction.ipynb
  - This will create prepared dara in the `processed` folder.
-  Run `python prediction.py` to obtain file of pkl format for predictions.
-  Run `python predictions.py` to obtain `predictions.csv` with predictions  on the test data.
