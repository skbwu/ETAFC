# Exploring Temporal Graph Neural Networks for Autoregressive Forecasting of COVID-19
This repo accompanies the paper "Exploring Temporal Graph Neural Networks for Autoregressive Forecasting of COVID-19" written by Skyler Wu '24 for Professor Melanie Weber's APMTH 220: Geometric Methods for Machine Learning, Spring 2024.

Raw data from the [Google COVID-19 Open Data Repository](https://health.google.com/covid-19/open-data/) as well shapefiles for map visualization from the [US Census](https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip) can be found in the `raw` folder, while processed data can be found in the `processed` folder (e.g., daily-aggregated-weekly data, central files for saving cleaned `location_key` and latitude/longitude data). The notebooks used to process the raw data and prototype some of the below pipelines can be found in `notebooks`.

Within the `scripts` folder:
- `linear_main.py` contains the main pipeline for running linear vector autoregression and standard autoregression models for COVID-19 disease forecasting.
- `dcrnn_main.py` contains the main pipeline for running DCRNN models for COVID-19 disease forecasting.
- `tgcn_main.py` contains the main pipeline for running T-GCN models for COVID-19 disease forecasting.

**To run all experiments on the FASRC Cannon high-performance computing cluster (or another SLURM-based cluster):** run `bash {linear, dcrnn, tgcn}_main_runscript_driver.sh` (select one of them) after making appropriate updates to the filepaths, fairshare accounts, and partition names, etc.

**To process raw experimental results and generate figures:** use `analyzer.ipynb` to generate aggregated log files and then use `main_visualizer.ipynb` to generate figures.
