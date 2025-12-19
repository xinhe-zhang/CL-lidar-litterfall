## CL-lidar-litterfall

The initial file structure should look like this:
```
CL_lidar-litterfall
|- source
|- spatial files
```

The spatial_files directory must include the following datasets, which are available for download [here](https://www.google.com/search?q=%E4%BD%A0%E7%9A%84%E9%9B%B2%E7%AB%AF%E9%80%A3%E7%B5%90):

```
spatial_files
|- CHM_2021_clip.tif
|- mL_map_2021.tif
|- mL_map_2022.tif
|- mL_map_2023.tif
|- CLM.dbf
|- CLM.prj
|- CLM.shp
|- CLM.shx
```
The project relies on the following core libraries for spatial data processing and GPU acceleration:
affine==2.4.0
attrs==25.3.0
certifi==2025.8.3
click==8.3.0
click-plugins==1.1.1.2
cligj==0.7.2
cupy-cuda12x==13.6.0
fastrlock==0.8.3
geopandas==1.1.1
numpy==2.3.3
packaging==25.0
pandas==2.3.3
pyogrio==0.11.1
pyparsing==3.2.5
pyproj==3.7.2
python-dateutil==2.9.0.post0
pytz==2025.2
rasterio==1.4.3
shapely==2.1.2
six==1.17.0
tqdm==4.67.1
tzdata==2025.2


