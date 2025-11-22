import copy
import pandas as pd


def optimize_reg_coef(model_class, starts, stops, errors, params, max_iter=50, max_successful_runs=5):
    
    best_model = None
    low, high = 0, 1
    successful_runs = 0
    
    for i in range(0, max_iter):
        mid = (low + high) / 2
        new_model = model_class(starts, stops, errors, **params)
        new_model.reg_coef = mid
        
        try:
            new_model.run()
            
        except Exception as error:
            high = mid
            
        else:
            if (new_model.weights == 0).all():
                continue
                
            low = mid           
            successful_runs += 1
            
            if best_model is None or (new_model.weights != 0).sum().item() < (best_model.weights != 0).sum().item():
                best_model = copy.deepcopy(new_model)
        
        if successful_runs >= max_successful_runs:
            break
            
    if best_model is None:
        raise ValueError("model divergence", (stops - starts).max())
    
    return best_model


def make_df(model):
    if model.device != "cpu":
        model.to("cpu")
    df = pd.DataFrame({"start": model.starts.flatten(), "stop": model.stops.flatten()})
    df["dyads"] = model.dyads[model.Hij.argmax(1)]
    df["template|dyad"] = model.Hij.max(1)[0]
    return df