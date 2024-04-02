from sklearn_quantile import RandomForestQuantileRegressor
import config_param

default_random_state_base = 1

def get_models(num_models):
    models = [
        RandomForestQuantileRegressor(n_estimators=config_param.default_n_estimators, 
                                       random_state=default_random_state_base + i) 
        for i in range(num_models)
    ]
    return models
