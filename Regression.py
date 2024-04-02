import numpy as np
import pandas as pd
import argparse
from datetime import datetime, timedelta
import config_param
import utils
import config_model
from concurrent.futures import ProcessPoolExecutor, as_completed



# Initialize the parser
parser = argparse.ArgumentParser(description='Load CSV files and population data for regression analysis.')

# Add arguments for the CSV file paths 
parser.add_argument('pedili_old_path', type=str, help='Path to the Pedictors file')
parser.add_argument('hh_path', type=str, help='Path to the Hospitalizations file')
parser.add_argument('popu_path', type=str, help='Path to the us_states_population_data file')

# Parse the command line arguments
args = parser.parse_args()

# Load the CSV files using the provided paths
PedILI_Old = pd.read_csv(args.pedili_old_path, header=None)
H = pd.read_csv(args.hh_path, header=None)
popu = np.loadtxt(args.popu_path)
old_data_no2020 = PedILI_Old.copy() 
num_states = popu.shape[0]
days = H.shape[0] / num_states

st = pd.DataFrame()
extender = old_data_no2020.shape[0]//config_param.offset//num_states
for i in range(1, num_states+1):
    st = pd.concat([st, pd.DataFrame({'state_id': [i] * extender})], ignore_index=True)
#----   
st = pd.concat([st] * config_param.offset, ignore_index=True) 
old_data_no2020 = pd.concat([old_data_no2020, st.iloc[:len(old_data_no2020), :].reset_index(drop=True)], axis=1)
# Prepare 'st' for 'H' DataFrame, adjusting for 'days'
st_for_H = pd.DataFrame()
for i in range(1, num_states+1):
    st_for_H = pd.concat([st_for_H, pd.DataFrame({'state_id': [i] * int(days)})], ignore_index=True)  
H = pd.concat([H, st_for_H.iloc[:len(H), :].reset_index(drop=True)], axis=1)
#----
new_col_index = len(old_data_no2020.columns)
for i in range(len(popu)):
    #print(i)
    old_data_no2020.loc[old_data_no2020['state_id'] == i + 1, new_col_index] = popu[i]
    H.loc[H['state_id'] == i + 1, new_col_index] = popu[i]
#----
if 'state_id' in H.columns:
    H.drop(columns=['state_id'], inplace=True)
if 'state_id' in old_data_no2020.columns:
    old_data_no2020.drop(columns=['state_id'], inplace=True)
H2 = H.copy()
old_data_no20202 = old_data_no2020.copy()

last_col_index = H2.columns[-1]

for i in range(277):  
    H2.iloc[:, i] = (H2.iloc[:, i] * 100000) / H2[last_col_index]
    old_data_no20202.iloc[:, i] = (old_data_no20202.iloc[:, i] * 100000) / old_data_no20202[last_col_index]
#----
H2.columns.values[0] = 'wkbhnd'
H2.columns.values[1] = 'incwk'
old_data_no20202.columns.values[0] = 'wkbhnd'
old_data_no20202.columns.values[1] = 'incwk'

# For the pop column, which is the last column, rename using its index
H2.columns.values[-1] = 'pop'
old_data_no20202.columns.values[-1] = 'pop'

#---- TRAIN/TEST DATA PREPARATION

data_train = pd.DataFrame()
data_test = pd.DataFrame()

for i in range(int(H2.shape[0] / days)):
    start_idx = int(i * days)
    end_idx = int(days * (i + 1))
    days_int = int(days)  
    data_train = pd.concat([data_train, H2.iloc[start_idx:end_idx - 7 * (config_param.wks_back), :]], ignore_index=True)
    data_test = pd.concat([data_test, H2.iloc[end_idx - 7 * (config_param.wks_back ):end_idx - (7 * (config_param.wks_back - 1)), :]], ignore_index=True)

#---- WEIGHTS
    
total_weights = (len(data_train) + len(old_data_no20202)) // len(popu)
weights = np.zeros(total_weights)

num_weights = weights.shape[0]  # weights shape is (N,1)
weights = np.zeros(num_weights)

# Calculate the weights based on decay_factor for each week
for day_idx in range(1, num_weights + 1):
    wk = np.ceil(day_idx / 7)
    weights[day_idx - 1] = config_param.decay_factor ** wk

# Flip the weights array to reverse its order and ensure it's a column vector
weights = np.flip(weights).reshape(-1, 1)
weights = np.repeat(weights, len(popu), axis=0)

#---- TRAINING DATA PROCESSSING
col_increment = config_param.npredictors // config_param.num_dh_rates_sample #confirm

start_col = config_param.bin_size # confirm
end_col = start_col + col_increment
target_col_index = 2  # For the first model #confirm

# Dictionary to store the cleaned data for each model
cleaned_data = {}

# Loop for weeks_ahead --> partitioning training data for 1 week ahead, 2 week ahead ... so on
for model_num in range(1, config_param.weeks_ahead+1):
    X_clean, y_clean, weights_clean = utils.prepare_data_for_model(start_col, end_col, target_col_index, old_data_no20202, data_train, weights)
    cleaned_data[f"model_{model_num}"] = (X_clean, y_clean, weights_clean)
    
    # Update for next model
    start_col = end_col  # Next model starts where previous ended
    end_col += col_increment
    target_col_index += 1  # Move to the next target column
#               X         y        w
datasets = [(data[0], data[1], data[2]) for data in cleaned_data.values()]

#---- TRAINING DATA
num_models = config_param.weeks_ahead
models = config_model.get_models(num_models)
trained_models_dict = {}

with ProcessPoolExecutor(max_workers=5) as executor:
    # Create a dictionary to map futures to model names based on the cleaned_data keys
    future_to_name = {executor.submit(utils.train_model, model, *dataset): name for model, dataset, name in zip(models, datasets, cleaned_data.keys())}
    for future in as_completed(future_to_name):
        model_name = future_to_name[future]
        try:
            trained_models_dict[model_name] = future.result()
        except Exception as exc:
            print(f"{model_name} generated an exception: {exc}")


"""
Model saving
from joblib import dump,load

for name, model in trained_models_dict.items():
    print(name, model)
    print("saved", name, model)
    dump(model, f'{name}.joblib'

trained_models_dict={}

for i in range(1, num_models+1):
    #print(name, model)
    #print("saved", name, model)
    trained_models_dict[f'model_{i}'] = load(f'model_{i}.joblib')

"""
#---- PREDICTION

mpgQuartiles_dict = {}
mpgMean_dict = {}
pmf_dict = {}
quantiles = config_param.quantiles
# Define the start and end columns for the first model's predictors and adjust for each subsequent model
start_col = config_param.bin_size  # Starting column for the first model

# Assuming 'data_test' is defined and 'trained_models_dict' contains your models
# Assuming 'quantiles' is defined for predict_quantiles function
for model_num in range(1, num_models+1):
    model_key = f'model_{model_num}'
    end_col = start_col + col_increment - 1  # Adjusted for Python's zero-based indexing
    print(start_col, end_col+1)
    # Selecting the appropriate columns for X_test for the current model
    
    X_test = data_test.iloc[:, [0, 1] + list(range(start_col, end_col + 1)) + [-1]]
    X_test.columns = X_test.columns.astype(str)
    X_test = X_test.iloc[:, 2:]  # Drop the first two columns
    
    # Generate predictions using the functions and the current model
    mpgQuartiles = utils.predict_quantiles(trained_models_dict[model_key], X_test, quantiles)
    mpgMean = utils.getmean(trained_models_dict[model_key], X_test)
    
    # Adjust the index for pmf_class if required by your logic
    pmf_index = model_num - 1 if model_num < 5 else 2  # Adjust based on your specific logic #confirm
    pmf = utils.pmf_class(trained_models_dict[model_key], X_test, pmf_index)
    
    # Store the predictions, means, and PMFs in dictionaries
    mpgQuartiles_dict[model_key] = mpgQuartiles
    mpgMean_dict[model_key] = mpgMean
    pmf_dict[model_key] = pmf
    
    # Update start_col for the next model
    start_col = end_col + 1
#

#Parrallization
"""
# Assuming quantiles, data_test, and trained_models_dict are defined as before

start_col = config_param.bin_size

if __name__ == '__main__':
    mpgQuartiles_dict = {}
    mpgMean_dict = {}
    pmf_dict = {}
    error_messages = []

    with ProcessPoolExecutor(max_workers=1) as executor:
        futures = []
        for model_num in range(1, num_models + 1):
            model_key = f'model_{model_num}'
            end_col = start_col + col_increment - 1
            
            if end_col >= len(data_test.columns):
                print(f"Adjusting end_col from {end_col} to {len(data_test.columns)-1} for model {model_key}")
                end_col = len(data_test.columns) - 1

            future = executor.submit(utils.predict_modely, model_num, data_test, trained_models_dict, quantiles, start_col, end_col)
            futures.append(future)
            start_col = end_col + 1  # Prepare start_col for the next iteration

        for future in as_completed(futures):
                try:
                    result = future.result()  # Attempt to get the result of the task
                    print(result)  # Assuming this prints a tuple (model_key, mpgQuartiles, mpgMean, pmf, error)
                    
                    # Check if the last element of the tuple indicates an error
                    if result[-1]:  # Assuming the last item in the tuple is an error message or None
                        error_messages.append(result[-1])
                    else:
                        model_key, mpgQuartiles, mpgMean, pmf = result[:-1]
                        mpgQuartiles_dict[model_key] = mpgQuartiles
                        mpgMean_dict[model_key] = mpgMean
                        pmf_dict[model_key] = pmf
                except Exception as exc:
                    
                    error_message = f"A task generated an exception: {exc}"
                    print(error_message)
                    error_messages.append(error_message)
    # Handle potential errors
    if error_messages:
        for error in error_messages:
            print(error)
"""
#---- AGGREGATION

pred_dict = {}
pred_ms_dict = {}

# Loop through each model in the dictionaries
for model_key in mpgQuartiles_dict:
    # Retrieve the quartiles and means for the current model
    mpgQuartiles = mpgQuartiles_dict[model_key]
    mpgMean = mpgMean_dict[model_key]
    
    # Aggregate quartiles and means for the current model
    pred = utils.aggregate_quartiles(mpgQuartiles)
    pred_ms = utils.aggregate_means(mpgMean)
    
    # Store the aggregated data in new dictionaries
    pred_dict[model_key] = pred
    pred_ms_dict[model_key] = pred_ms

#---- OUTPUT FORMAT
thisday = days - config_param.bin_size*(config_param.wks_back-1)
    
quant_deaths = np.array([0.01, 0.025] + list(np.arange(0.05, 1.0, 0.05)) + [0.975, 0.99])

# Initialize the arrays for storing quantile and mean predictions
quant_preds_deaths = np.nan * np.zeros((num_states, config_param.weeks_ahead, quant_deaths.shape[0]))
mean_preds_deaths = np.nan * np.zeros((num_states, config_param.weeks_ahead))

# Fill the arrays with the respective data
for i in range(5):
    mean_preds_deaths[:, i] = pred_ms_dict[f"model_{i+1}"]
    quant_preds_deaths[:, i, :] = pred_dict[f"model_{i+1}"]

# Adjusting the predictions by population
for i in range(num_states):
    mean_preds_deaths[i, :] = mean_preds_deaths[i, :] * popu[i] / 100000
    quant_preds_deaths[i, :, :] = quant_preds_deaths[i, :, :] * popu[i] / 100000

#----
    
fips_tab = pd.read_csv('reich_fips.txt')
abvs = pd.read_csv('us_states_abbr_list.txt', header=None)[0].tolist()

fips = [fips_tab.loc[fips_tab['abbreviation'] == abv, 'location'].values[0] for abv in abvs]

# Prepare the table Ti for quantile predictions
thesevals = quant_preds_deaths.flatten()
cid, wh, qq = np.unravel_index(range(len(thesevals)), quant_preds_deaths.shape)
wh_string = np.arange(1, max(wh) + 1).astype(str)

Ti = pd.DataFrame({
    'reference_date': [(config_param.zero_date + timedelta(days=thisday + config_param.bin_size)).strftime('%Y-%m-%d')] * len(thesevals),
    'horizon': wh - 2,
    'target': ['wk inc flu hosp'] * len(thesevals),
    'target_end_date': [(config_param.zero_date + timedelta(days=thisday + config_param.bin_size + 7 * (w - 2))).strftime('%Y-%m-%d') for w in wh],
    'location': [fips[c] for c in cid],
    'output_type': ['quantile'] * len(thesevals),
    'output_type_id': quant_deaths[qq],
    'value': np.round(thesevals, 1).astype(str)
})

# Prepare the table Tm for pmf predictions
temp = pmf[0:num_states, :, :]
idx = np.argmax(pmf, axis=1, keepdims=True)
thesevals = temp.flatten()
cid, class_, wh = np.unravel_index(range(len(thesevals)), pmf.shape)
classes = ["stable", "increase", "large_increase", "decrease", "large_decrease"]

Tm = pd.DataFrame({
    'reference_date': [(config_param.zero_date + timedelta(days=thisday + config_param.bin_size)).strftime('%Y-%m-%d')] * len(thesevals),
    'horizon': wh - 2,
    'target': ['wk flu hosp rate change'] * len(thesevals),
    'target_end_date': [(config_param.zero_date + timedelta(days=thisday + config_param.bin_size + 7 * (w - 2))).strftime('%Y-%m-%d') for w in wh],
    'location': [fips[c] for c in cid],
    'output_type': ['pmf'] * len(thesevals),
    'output_type_id': [classes[c] for c in class_],
    'value': np.round(thesevals, 1).astype(str)
})

# Combine Ti and Tm
T_all = pd.concat([Ti, Tm], ignore_index=True)

#----


# Assuming quant_preds_deaths, pmf, and other required variables like zero_date, thisday, offset, quant_deaths, classes are defined

# Sum quant_preds_deaths across the first dimension
us_quants = np.sum(quant_preds_deaths, axis=1).flatten()
thesevals = us_quants

# Create a DataFrame for Ti
reference_date = [(config_param.zero_date + timedelta(days=thisday + config_param.bin_size)).strftime('%Y-%m-%d')] * len(thesevals)
horizon = np.arange(len(thesevals)) % 5 - 2  # Assuming quant_preds_deaths.shape[1] == 5 for horizon calculation
target = ['wk inc flu hosp'] * len(thesevals)
target_end_date = [(config_param.zero_date + timedelta(days=thisday + config_param.bin_size + 7 * (h + 2))).strftime('%Y-%m-%d') for h in horizon]
location = ['US'] * len(thesevals)
output_type = ['quantile'] * len(thesevals)
output_type_id = [quant_deaths[i % len(quant_deaths)] for i in range(len(thesevals))]
value = np.round(thesevals, 1).astype(str)

Ti = pd.DataFrame({
    'reference_date': reference_date,
    'horizon': horizon,
    'target': target,
    'target_end_date': target_end_date,
    'location': location,
    'output_type': output_type,
    'output_type_id': output_type_id,
    'value': value
})

# Create a DataFrame for Tm
temp = pmf[-1, :, :].flatten()
thesevals = temp

# Assuming classes is a list or array-like of class names
output_type_id = [classes[i % len(classes)] for i in range(temp.shape[0])]
Tm = pd.DataFrame({
    'reference_date': reference_date[:len(thesevals)],  # Reuse the reference_date calculation
    'horizon': horizon[:len(thesevals)],
    'target': ['wk flu hosp rate change'] * len(thesevals),
    'target_end_date': target_end_date[:len(thesevals)],
    'location': ['US'] * len(thesevals),
    'output_type': ['pmf'] * len(thesevals),
    'output_type_id': output_type_id,
    'value': np.round(thesevals, 1).astype(str)
})

# Concatenate Ti and Tm into T_all
if 'T_all' not in locals():
    T_all = pd.DataFrame()  # Initialize if T_all doesn't exist
T_all = pd.concat([T_all, Ti, Tm], ignore_index=True)

# Remove rows based on location
bad_idx = T_all['location'].isin(['60', '66', '69', '78'])
T_all = T_all[~bad_idx]

# Define path and filename
pathname = './24_forecasts/'
thisdate = (config_param.zero_date + timedelta(days=thisday + config_param.bin_size)).strftime('%Y-%m-%d')
filename = f'{pathname}{thisdate}-SGroup-RandomForest.csv'