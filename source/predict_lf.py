import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
import time
import cupy as cp
import gc
from tqdm import tqdm
import sys
import traceback

# --- parameters ---
CHUNK_SIZE = 100000
GPU_SUB_CHUNK_SIZE = 15000
POSTERIOR_BATCH_SIZE = 30000 


if __name__ == '__main__':

    # ML raster files (mL is the input variable)
    RASTER_FILES_ML = {
        2021: "../spatial files/mL_map_2021.tif",
        2022: "../spatial files/mL_map_2022.tif",
        2023: "../spatial files/mL_map_2023.tif"
    }
    
    # Radiation anomalies (ERA5 anom)
    ANOM_VALUES_PER_YEAR = {
        2021: 0.114558259,  
        2022: -0.114476411, 
        2023: -0.022922132 
    }

    # Posteriors from R (brms model output)
    POSTERIOR_SAMPLES_FILE = "./posterior_samples_fit4_with_SW.csv"
    
    

    print(">>> (MAIN) Preparing output directory...")
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f">>> Output directory '{OUTPUT_DIR}' is ready.")

    print(">>> (MAIN) Importing brms model posterior samples...")
    try:
        posterior_samples = pd.read_csv(POSTERIOR_SAMPLES_FILE)
        n_posterior_samples = len(posterior_samples)
        print(f">>> Successfully imported {n_posterior_samples} posterior samples.")
    except FileNotFoundError:
        print(f"Error: Posterior sample file '{POSTERIOR_SAMPLES_FILE}' does not exist. Please check the path.")
        sys.exit(1)

    print(">>> (MAIN) Transporting model parameters to GPU...")
    try:
        # 'b_b_Intercept' (population level b)
        b_b_gpu = cp.asarray(posterior_samples['b_b_Intercept'].values, dtype=cp.float32)
        
        # 'b_trL_Intercept' (population level trL)
        b_trl_intercept_gpu = cp.asarray(posterior_samples['b_trL_Intercept'].values, dtype=cp.float32)

        # Radiation anomaly's effect on trL
        col_name_anom = 'b_trL_anom' 
        if col_name_anom in posterior_samples.columns:
            b_trL_anom_gpu = cp.asarray(posterior_samples[col_name_anom].values, dtype=cp.float32)
            print(f">>> Successfully loaded '{col_name_anom}' parameter.")
        else:
            print(f"Error: Cannot find required column '{col_name_anom}'.")
            print(">>> Detected 'fit4_with_SW' model. Please ensure the R-exported column name is 'b_trL_anom'.")
            sys.exit(1)
            
    except KeyError as e:
        print(f"Error: Cannot find required column: {e}")
        print(">>> Please ensure the parameter names for 'formula_with_SW' (b, trL, anom) are correct.")
        sys.exit(1)

    print(">>> Successfully loaded model parameters into GPU.")
    del posterior_samples
    gc.collect()

    overall_start_time = time.time()

    # Loop through RASTER_FILES_ML
    for year, ml_raster_file in RASTER_FILES_ML.items():
        print(f"\n{'='*60}")
        print(f">>> Starting processing for year {year}")
        print(f"{'='*60}")

        # Output folder
        OUTPUT_DIR = Path(f"../fpredict_litterfall_chunks_{year}")

        # Start timer for the year
        year_start_time = time.time()

        # Check if 'anom' exists
        if year not in ANOM_VALUES_PER_YEAR:
            print(f"Error: 'anom' value for year {year} not found in ANOM_VALUES_PER_YEAR dictionary.")
            continue
        
        # Get the 'ERA5 anom' value for the year
        anom_value_for_year = ANOM_VALUES_PER_YEAR[year]
        print(f">>> Using anom value for year {year} (ERA5 anom): {anom_value_for_year}")

        # Check if the zero/NA file already exists in the output folder
        zero_output_file = OUTPUT_DIR / f"pred_{year}_zeros_and_na.csv"
        zero_file_exists = zero_output_file.exists()
        
        if zero_file_exists:
            print(f">>> Detected that '{zero_output_file}' already exists. Skipping processing for no-data/zero-value cells.")

        print(f">>> Reading raster data for year {year}: '{ml_raster_file}'...")
        try:
            # Read only the mL raster band
            with rasterio.open(ml_raster_file) as src:
                raster_data = src.read(1, masked=True)
                # np.ma.filled() returns input as an ndarray, with masked values replaced by fill_value (np.nan here).
                all_vals = np.ma.filled(raster_data, np.nan).flatten().astype(np.float32) 
                df = pd.DataFrame({'mL': all_vals, 'cell_id': np.arange(len(all_vals))})
                
        except rasterio.errors.RasterioIOError as e:
            print(f"Error: Failed to read raster file: {e}. Please check if the file exists and the format is correct.")
            continue

        print(f">>> Data loaded for year {year}, total {len(df)} pixels.")
        del raster_data, all_vals
        gc.collect()

        if not zero_file_exists:
            print(">>> Separating valid data from no-data/zero-value data...")
            is_invalid_or_zero = df['mL'].isna() | (df['mL'] == 0)
            df_zero_output = df[is_invalid_or_zero]
            df_predict = df[~is_invalid_or_zero].reset_index(drop=True) # The "~" operator means "NOT"

            if not df_zero_output.empty:
                print(f">>> Detected {len(df_zero_output)} no-data or zero-value pixels.")
                
                result_zero_df = pd.DataFrame({
                    'year': year, 'cell_id': df_zero_output['cell_id'],
                    'q025': 0.0, 'q50': 0.0, 'q975': 0.0,
                    'sd': 0.0, 'cv': 0.0 
                })
                result_zero_df.to_csv(zero_output_file, index=False)
                print(f">>> No-data/zero-value results saved to '{zero_output_file}'.")
                del result_zero_df, df_zero_output
            else:
                print(">>> No no-data or zero-value pixels found.")
        else:
            print(">>> Filtering for valid data...")
            is_invalid_or_zero = df['mL'].isna() | (df['mL'] == 0)
            df_predict = df[~is_invalid_or_zero].reset_index(drop=True)
            print(f">>> Filtering complete. {len(df_predict)} valid pixels require GPU computation.")

        del df
        gc.collect()

        if df_predict.empty:
            print(f">>> No valid data for year {year} requires GPU computation. Skipping.")
            continue

        total_chunks = (len(df_predict) - 1) // CHUNK_SIZE + 1
        print(f">>> Valid data for year {year}: {len(df_predict)} pixels, divided into {total_chunks} chunks.")

        # Check for progress / completed chunks
        completed_chunks = set()
        for f in OUTPUT_DIR.glob(f"pred_{year}_*.csv"):
            parts = f.stem.split('_')
            # Check if filename is of format 'pred_YEAR_#####.csv'
            if len(parts) == 3 and parts[2].isdigit(): 
                completed_chunks.add(int(parts[2]))

        if completed_chunks:
            print(f">>> Detected {len(completed_chunks)} completed chunks. Skipping these chunks.")
        else:
            print(">>> No completed progress found. Starting from scratch.")

        print(f">>> Pre-calculating trL parameter for year {year}...")
        # trL = b_trL_Intercept + anom * b_trL_anom
        trL_gpu = b_trl_intercept_gpu + (anom_value_for_year * b_trL_anom_gpu)
        print(">>> trL parameter calculation complete.")

        for i in tqdm(range(1, total_chunks + 1), desc=f"Processing Year {year}", unit="chunk"):
            if i in completed_chunks:
                continue

            try:
                start_index = (i - 1) * CHUNK_SIZE
                end_index = min(i * CHUNK_SIZE, len(df_predict))
                chunk_data = df_predict.iloc[start_index:end_index]

                if chunk_data.empty:
                    continue

                final_q025_cpu, final_q50_cpu, final_q975_cpu = [], [], []
                final_sd_cpu, final_cv_cpu = [], []
                n_pixels_in_chunk = len(chunk_data)
                
                n_sub_chunks = (n_pixels_in_chunk - 1) // GPU_SUB_CHUNK_SIZE + 1

                for j in range(n_sub_chunks):
                    sub_start = j * GPU_SUB_CHUNK_SIZE
                    sub_end = min((j + 1) * GPU_SUB_CHUNK_SIZE, n_pixels_in_chunk)
                    sub_chunk_pixels = chunk_data.iloc[sub_start:sub_end]

                    # Only load mL to GPU
                    ml_gpu = cp.asarray(sub_chunk_pixels['mL'].values, dtype=cp.float32)
                    
                    all_predictions_gpu = []
                    n_batches = (n_posterior_samples - 1) // POSTERIOR_BATCH_SIZE + 1
                    
                    for batch_idx in range(n_batches):
                        batch_start = batch_idx * POSTERIOR_BATCH_SIZE
                        batch_end = min(batch_start + POSTERIOR_BATCH_SIZE, n_posterior_samples)
                        
                        # Get the pre-calculated trL batch
                        trL_batch = trL_gpu[batch_start:batch_end]
                        b_b_batch = b_b_gpu[batch_start:batch_end]
                        
                        # The prediction formula:
                        # litterfall_sum = exp(mL * (trL_Intercept + anom * b_trL_anom) + b_b_Intercept)
                        # (n_pixels, 1) * (n_post_batch,) -> (n_pixels, n_post_batch)
                        pred_batch_gpu = cp.exp(ml_gpu[:, cp.newaxis] * trL_batch + b_b_batch)
                        
                        all_predictions_gpu.append(pred_batch_gpu)
                        
                        del pred_batch_gpu, trL_batch, b_b_batch
                    
                    del ml_gpu 
                    cp.get_default_memory_pool().free_all_blocks()

                    pred_matrix_gpu = cp.concatenate(all_predictions_gpu, axis=1)
                    del all_predictions_gpu

                    # ----------------- Calculate Metrics on GPU -----------------
                    # 1. Calculate quantiles
                    q025_gpu = cp.percentile(pred_matrix_gpu, 2.5, axis=1)
                    q50_gpu = cp.percentile(pred_matrix_gpu, 50, axis=1)
                    q975_gpu = cp.percentile(pred_matrix_gpu, 97.5, axis=1)
                    
                    # 2. Calculate SD
                    sd_gpu = cp.std(pred_matrix_gpu, axis=1)
                    
                    # 3. Calculate CV (sd / mean)
                    mean_gpu = cp.mean(pred_matrix_gpu, axis=1)
                    cv_gpu = cp.full_like(mean_gpu, cp.nan) # Initialize as nan
                    mask_cv_gpu = (mean_gpu != 0) # Build mask to prevent division by zero
                    cv_gpu[mask_cv_gpu] = sd_gpu[mask_cv_gpu] / mean_gpu[mask_cv_gpu]
                    # ----------------------------------------------------

                    # 4. Release large GPU items
                    del pred_matrix_gpu, mean_gpu, mask_cv_gpu 

                    # 5. Send everything back to CPU
                    final_q025_cpu.append(cp.asnumpy(q025_gpu))
                    final_q50_cpu.append(cp.asnumpy(q50_gpu))
                    final_q975_cpu.append(cp.asnumpy(q975_gpu))
                    final_sd_cpu.append(cp.asnumpy(sd_gpu)) 
                    final_cv_cpu.append(cp.asnumpy(cv_gpu)) 

                    # 6. Release GPU memory
                    del q025_gpu, q50_gpu, q975_gpu, sd_gpu, cv_gpu 
                    cp.get_default_memory_pool().free_all_blocks()

                # Consolidate results for the entire chunk
                result_df = pd.DataFrame({
                    'year': year,
                    'cell_id': chunk_data['cell_id'].values,
                    'q025': np.concatenate(final_q025_cpu),
                    'q50': np.concatenate(final_q50_cpu),
                    'q975': np.concatenate(final_q975_cpu),
                    'sd': np.concatenate(final_sd_cpu), 
                    'cv': np.concatenate(final_cv_cpu)  
                })

                output_file = OUTPUT_DIR / f"pred_{year}_{i:05d}.csv"
                result_df.to_csv(output_file, index=False)
                del chunk_data, result_df
                gc.collect()

            except Exception as e:
                print(f"\nError: Encountered a serious error while processing year {year} chunk {i}: {e}")
                print("Detailed error traceback:")
                traceback.print_exc()
                print(f"Skipping chunk {i} and continuing...")
                continue

        # Release resources for the year
        del df_predict, trL_gpu
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

        year_end_time = time.time()
        print(f"\n>>> Year {year} processing complete! Time elapsed: {(year_end_time - year_start_time) / 60:.2f} minutes")

    # Release main parameters on GPU
    del b_b_gpu, b_trl_intercept_gpu, b_trL_anom_gpu
    cp.get_default_memory_pool().free_all_blocks()

    overall_end_time = time.time()
    print(f"\n{'='*60}")
    print(f">>> All years processed successfully!")
    print(f">>> Total time elapsed: {(overall_end_time - overall_start_time) / 60:.2f} minutes")
    print(f">>> Output directory: {OUTPUT_DIR}")
    print(f">>> Reminder: You need to merge all the .csv files in '{OUTPUT_DIR}' for final results.")
    print(f"{'='*60}")