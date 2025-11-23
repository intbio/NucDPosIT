import copy
import pandas as pd


# def train_model(starts,
#     ends,
#     errors,
#     params,
#     dyad_dist,
#     max_iter,
#     device,
#     train_iter,
#     max_successful_trains,
#     nretries,):
#     try:
#         best_model = optimize_model(starts,
#     ends,
#     errors,
#     params,
#     dyad_dist,
#     max_iter,
#     device,
#     train_iter,
#     max_successful_trains,
#     nretries,)


# # def optimize_reg_coef(model_class, starts, stops, errors, params, max_iter=50, max_successful_runs=5):
    
# #     best_model = None
# #     low, high = 0, 1
# #     successful_runs = 0
    
# #     for i in range(0, max_iter):
# #         mid = (low + high) / 2
# #         new_model = model_class(starts, stops, errors, **params)
# #         new_model.reg_coef = mid
        
# #         try:
# #             new_model.run()
            
# #         except Exception as error:
# #             high = mid
            
# #         else:
# #             if (new_model.weights == 0).all():
# #                 continue
                
# #             low = mid           
# #             successful_runs += 1
            
# #             if best_model is None or (new_model.weights != 0).sum().item() < (best_model.weights != 0).sum().item():
# #                 best_model = copy.deepcopy(new_model)
        
# #         if successful_runs >= max_successful_runs:
# #             break
            
# #     if best_model is None:
# #         raise ValueError("model divergence", (stops - starts).max())
    
# #     return best_model


# def optimize_model(
#     starts,
#     ends,
#     errors,
#     params,
#     dyad_dist,
#     max_iter,
#     device,
#     train_iter,
#     max_successful_trains,
#     nretries,
# ):
#     try:
#         best_model = optimize_reg_coef(
#             StochasticEMMOdel,
#             starts,
#             ends,
#             errors,
#             {
#                 "dyad_dist": dyad_dist,
#                 "max_iter": max_iter,
#                 "device": device,
#             },
#             train_iter,
#             max_successful_trains,
#         )
#         print(
#             f"Window {idx} on chromosome {chromosome}: dyads: {best_model.dyads.shape[0]}"
#         )
#         success = True

#     except Exception as error:
#         print(f"Window {idx} on chromosome {chromosome}: ERROR - {error}, RETRYING")

#         # Retry with reduced dyad distance
#         for i in range(nretries):
#             new_dyad_dist = dyad_dist // 2 ** (i + 1)
#             if new_dyad_dist == 0:
#                 new_dyad_dist = 1

#             try:
#                 best_model = optimize_reg_coef(
#                     StochasticEMMOdel,
#                     starts,
#                     ends,
#                     errors,
#                     {
#                         "dyad_dist": new_dyad_dist,
#                         "max_iter": max_iter,
#                         "device": device,
#                     },
#                     train_iter,
#                     max_successful_trains,
#                 )
#                 success = True
#                 print(f"Success on retry {i+1} with dyad_dist={new_dyad_dist}")
#                 break

#             except Exception as retry_error:
#                 print(f"Retry {i+1} failed: {retry_error}")
#                 continue

#     if not success or best_model is None:
#         print(f"ERROR: Skipping window {idx} on chromosome {chromosome}")
#         raise ValueError("empty model")
#     return best_model


# def process_optimized_model(best_model, window_size, step, idx, chromosome, df_path, dyads_df_path):
#     best_model.to('cpu')

#     df = make_df(best_model)
#     df['chr'] = chromosome
#     df['n'] = idx

#     if idx != 0:               
#         dyad_thold = window_size + step * (idx - 1) - 200
#         df = df.query("dyads >= @dyad_thold")
#     else:
#         dyad_thold = window_size - 200
#         df = df.query("dyads <= @dyad_thold")

#     # Create dyads BED file entries
#     dyads_bed = df.groupby('dyads', as_index=False).size()
#     dyads_bed['chr'] = chromosome  # Fixed: was hardcoded to "NC_001136.10"
#     dyads_bed['stop'] = dyads_bed['dyads'] + 1
#     # dyads_bed.rename(columns={'dyads': 'start', 'size': 'score'}, inplace=True)

#     # Append to output files
#     with open(df_path, 'a') as df_file, open(dyads_df_path, 'a') as dyads_file:
#         df.to_csv(df_file, index=False, header=False, mode='a')
#         dyads_bed[['chr', 'start', 'stop', 'score']].to_csv(
#             dyads_file, index=False, header=False, sep='\t', mode='a'
#         )



def make_df(model):
    if model.device != "cpu":
        model.to("cpu")
    df = pd.DataFrame({"start": model.starts.flatten(), "stop": model.stops.flatten()})
    df["dyads"] = model.dyads[model.Hij.argmax(1)]
    df["template|dyad"] = model.Hij.max(1)[0]
    return df