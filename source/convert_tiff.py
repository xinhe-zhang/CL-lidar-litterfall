import pandas as pd
import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm
import glob
import gc
import os
import pyproj
import sys # <<< Keep sys module import

# Set PROJ_LIB environment variable to fix proj.db not found issue ---
try:
    proj_data_dir = pyproj.datadir.get_data_dir()
    os.environ['PROJ_LIB'] = proj_data_dir
    print(f">>> Successfully set PROJ data path: {proj_data_dir}")
except Exception as e:
    print(f"Warning: Failed to set PROJ_LIB automatically: {e}")
    print(">>> If coordinate transforms continue to fail, try setting the PROJ_LIB environment variable manually.")

# --- 1. Get and validate year parameter ---
# <<< MODIFIED: Use input() to prompt user for year >>>
while True:
    try:
        year_input = input("Please enter the year you want to process (e.g., 2023): ")
        YEAR = int(year_input)
        print(f">>> Setting processing year to: {YEAR}")
        break # Valid input, exit loop
    except ValueError:
        print(f"Error: '{year_input}' is not a valid integer. Please try again.")
    except (EOFError, KeyboardInterrupt):
        print("\n>>> Operation cancelled.")
        exit()

# --- 1b. Set parameters ---

# <<< ADD: Set your Shapefile path >>>
SHAPEFILE_PATH = "../spatial files/CLM.shp" 

# <<< ADD: Set CHM mask image path >>>
# !!! This image will be used to mask the output result !!!
CHM_RASTER_FILE = "../spatial files/CHM_2021_clip.tif" 

# --- Dynamically generate file and directory names ---
SOURCE_RASTER_FILE = f"../spatial files/mL_map_{YEAR}.tif" 

# <<< MODIFIED: Ensure directory name matches the first script >>>
# <<< MODIFIED: Dynamically use year >>>
PREDICTION_DIR = Path(f"../predict_litterfall_chunks_{YEAR}") 

OUTPUT_GEOTIFF_DIR = Path("../geotiff_results") 
BATCH_SIZE = 100
# ===================================================================

OUTPUT_GEOTIFF_DIR.mkdir(exist_ok=True)

# --- Check if input files exist ---
if not os.path.exists(SHAPEFILE_PATH):
    print(f"Error: Shapefile '{SHAPEFILE_PATH}' not found! Please check the path.")
    exit()
if not os.path.exists(SOURCE_RASTER_FILE):
    print(f"Error: Source Raster file '{SOURCE_RASTER_FILE}' not found!")
    print(f"       (Check if file for year {YEAR} exists)")
    exit()
if not os.path.exists(CHM_RASTER_FILE):
    print(f"Error: CHM mask file '{CHM_RASTER_FILE}' not found!")
    exit()

print(f">>> Processing year: {YEAR}")
print(">>> Step 1/7: Finding all prediction result CSV files...")

# <<< MODIFIED: Update search path to match year >>>
csv_files = sorted(glob.glob(str(PREDICTION_DIR / f"pred_{YEAR}_*.csv")))

if not csv_files:
    print(f"Error: No 'pred_{YEAR}_*.csv' files found in '{PREDICTION_DIR}'")
    exit()
print(f">>> Found {len(csv_files)} CSV files for year {YEAR}")

# --- 2. Prepare Raster template ---
print(">>> Step 2/7: Reading original Raster metadata...")
with rasterio.open(SOURCE_RASTER_FILE) as src:
    meta = src.meta.copy()
    height = src.height
    width = src.width
    n_total_cells = height * width
    shape = (height, width)
    source_crs = src.crs 
    
    meta.update(
        dtype=rasterio.float32,
        count=1,
        nodata=np.nan
    )

# --- 3. Read CHM image and create mask ---
print(f">>> Step 3/7: Reading CHM mask image '{CHM_RASTER_FILE}'...")
with rasterio.open(CHM_RASTER_FILE) as chm_src:
    # Check if dimensions match
    if chm_src.height != height or chm_src.width != width:
        print("="*60)
        print("Error: Dimension mismatch!")
        print(f"Source image '{SOURCE_RASTER_FILE}' dimensions: ({height}, {width})")
        print(f"CHM image '{CHM_RASTER_FILE}' dimensions: ({chm_src.height}, {chm_src.width})")
        print("Both images must have the exact same width and height to be masked.")
        print("="*60)
        exit()

    chm_data = chm_src.read(1)
    chm_mask = np.isnan(chm_data) | (chm_data == -9999)
    print(">>> CHM mask created successfully.")
    del chm_data 

# --- 4. Read and prepare Shapefile ---
print(f">>> Step 4/7: Reading Shapefile '{SHAPEFILE_PATH}'...")
gdf = gpd.read_file(SHAPEFILE_PATH)

if gdf.crs != source_crs:
    print(f"Warning: Shapefile CRS ({gdf.crs}) differs from Raster CRS ({source_crs}).")
    print(">>> Reprojecting Shapefile CRS to match Raster...")
    gdf = gdf.to_crs(source_crs)
    print(">>> CRS re-projection complete.")
clip_geoms = gdf.geometry.values

# --- 5. Use Memory-mapped files ---
print(">>> Step 5/7: Creating memory-mapped files...")
temp_dir = Path("../temp_arrays")
temp_dir.mkdir(exist_ok=True)

q025_mmap = np.memmap(temp_dir / 'q025.dat', dtype=np.float32, mode='w+', shape=(n_total_cells,))
q50_mmap = np.memmap(temp_dir / 'q50.dat', dtype=np.float32, mode='w+', shape=(n_total_cells,))
q975_mmap = np.memmap(temp_dir / 'q975.dat', dtype=np.float32, mode='w+', shape=(n_total_cells,))
sd_mmap = np.memmap(temp_dir / 'sd.dat', dtype=np.float32, mode='w+', shape=(n_total_cells,))
cv_mmap = np.memmap(temp_dir / 'cv.dat', dtype=np.float32, mode='w+', shape=(n_total_cells,))
q025_mmap[:] = np.nan
q50_mmap[:] = np.nan
q975_mmap[:] = np.nan
sd_mmap[:] = np.nan
cv_mmap[:] = np.nan


# --- 6. Batch process CSV files ---
print(f">>> Step 6/7: Starting batch processing (Batch size: {BATCH_SIZE} files)...")

total_batches = (len(csv_files) + BATCH_SIZE - 1) // BATCH_SIZE
processed_cells = 0

for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
    batch_start = batch_idx * BATCH_SIZE
    batch_end = min(batch_start + BATCH_SIZE, len(csv_files))
    batch_files = csv_files[batch_start:batch_end]
    
    for csv_file in batch_files:
        try:
            df = pd.read_csv(csv_file)
            
            # --- Core modification start ---
            # Check if necessary columns exist
            required_cols = ['cell_id', 'q025', 'q50', 'q975', 'sd', 'cv']
            if not all(col in df.columns for col in required_cols):
                print(f"\nWarning: File {csv_file} is missing required columns.")
                print(f"         (Requires: {required_cols})")
                print(">>>       Please ensure you used the *modified* GPU script (main.py) to generate CSVs.")
                print(">>>       Skipping this file.")
                continue
            
            cell_ids = df['cell_id'].values
            
            # --- Estimation removed ---
            
            # --- Direct read ---
            q025_mmap[cell_ids] = df['q025'].values
            q50_mmap[cell_ids] = df['q50'].values
            q975_mmap[cell_ids] = df['q975'].values
            sd_mmap[cell_ids] = df['sd'].values
            cv_mmap[cell_ids] = df['cv'].values
            
            # --- Core modification end ---
            
            processed_cells += len(df)
            del df
            
        except Exception as e:
            print(f"Warning: Error processing {csv_file}: {e}")
            continue
    
    q025_mmap.flush()
    q50_mmap.flush()
    q975_mmap.flush()
    sd_mmap.flush()
    cv_mmap.flush()
    gc.collect()

print(f">>> Processed {processed_cells:,} valid predictions")


# --- 7. Mask, clip, and write GeoTIFF ---
print(">>> Step 7/7: Masking, clipping, and writing GeoTIFF files...")

def process_and_write_geotiff(mmap_array, output_path, source_meta, source_shape, clip_geometries, chm_mask_array):
    """
    Masks, clips, and writes data from a memory-mapped array to a GeoTIFF file.
    """
    print(f"Processing: {output_path.name}")
    
    # 1. Reshape 1D mmap array to 2D image
    full_data = np.array(mmap_array).reshape(source_shape)
    
    # Apply CHM mask
    print("       - Applying CHM mask...")
    full_data[chm_mask_array] = 0 # (Set NaN/Nodata areas to 0)
    
    # 2. Use rasterio MemoryFile for in-memory operations
    print("       - Clipping image...")
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**source_meta) as dataset:
            dataset.write(full_data.astype(source_meta['dtype']), 1)
        
        with memfile.open() as dataset:
            out_image, out_transform = mask(
                dataset=dataset, 
                shapes=clip_geometries, 
                crop=True,
                nodata=source_meta['nodata'] # (Use 0 as nodata value)
            )
            out_meta = dataset.meta.copy()

    # 3. Update metadata
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

    # 4. Write final GeoTIFF file
    print("       - Writing file...")
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(out_image)
    print(f"       - Done: {output_path.name}")
    del full_data, out_image # Release memory

# Dynamically generate output file names
output_files = {
    'q025': OUTPUT_GEOTIFF_DIR / f"prediction_q025_{YEAR}_masked_clipped.tif",
    'q50': OUTPUT_GEOTIFF_DIR / f"prediction_q50_{YEAR}_masked_clipped.tif",
    'q975': OUTPUT_GEOTIFF_DIR / f"prediction_q975_{YEAR}_masked_clipped.tif",
    'sd': OUTPUT_GEOTIFF_DIR / f"prediction_sd_{YEAR}_masked_clipped.tif",
    'cv': OUTPUT_GEOTIFF_DIR / f"prediction_cv_{YEAR}_masked_clipped.tif"
}

# Call function, passing chm_mask
process_and_write_geotiff(q025_mmap, output_files['q025'], meta, shape, clip_geoms, chm_mask)
process_and_write_geotiff(q50_mmap, output_files['q50'], meta, shape, clip_geoms, chm_mask)
process_and_write_geotiff(q975_mmap, output_files['q975'], meta, shape, clip_geoms, chm_mask)
process_and_write_geotiff(sd_mmap, output_files['sd'], meta, shape, clip_geoms, chm_mask)
process_and_write_geotiff(cv_mmap, output_files['cv'], meta, shape, clip_geoms, chm_mask)


# --- 8. Clean up temporary files ---
print(">>> Cleaning up temporary files...")
del q025_mmap, q50_mmap, q975_mmap, sd_mmap, cv_mmap, chm_mask
gc.collect()

try:
    for f in temp_dir.glob('*.dat'):
        f.unlink()
    temp_dir.rmdir()
    print(">>> Temporary files cleaned up.")
except OSError as e:
    print(f"Warning: Error cleaning up temporary file {f}: {e}")


print("\n>>> All tasks complete!")
print(f">>> Masked and clipped output files are located in: {OUTPUT_GEOTIFF_DIR}/")