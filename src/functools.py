import copy
import pandas as pd


def optimize_reg_coef(model, max_iter=30, max_successful_runs=5):
    best_model = copy.deepcopy(model)
    low, high = 0, 1
    best_nonzero = (best_model.weights > 0).sum().item()
    successful_runs = 0
    
    for i in range(max_iter):
        mid = (low + high) / 2
        
        new_model = copy.deepcopy(model)
        new_model.reg_coef = mid
        
        try:
            new_model.run()
            successful_runs += 1
            current_nonzero = (new_model.weights > 0).sum().item()
            
            print(f"Iter {i}: reg_coef={mid:.4f}, components={current_nonzero}")
            
            if current_nonzero > 0:
                if current_nonzero < best_nonzero:
                    best_model = copy.deepcopy(new_model)
                    best_nonzero = (best_model.weights > 0).sum().item()

                        
                low = mid
            else:
                high = mid
            
            if successful_runs >= max_successful_runs:
                break
                
        except Exception as error:
            high = mid
    
    return best_model


def make_df(model):
    if model.device != "cpu":
        model.to("cpu")
    df = pd.DataFrame({"start": model.starts.flatten(), "stop": model.stops.flatten()})
    df["dyads"] = model.dyads[model.Hij.argmax(1)]
    df["template|dyad"] = model.Hij.max(1)[0]
    return df