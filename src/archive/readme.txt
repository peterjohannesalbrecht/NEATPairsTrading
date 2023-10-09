README - REPLICATING RESULTS

1. Software Architecture
- All core modules written and used for this thesis are in the src folder
- All scripts that produce (intermediary) results are in the top-level directory
- Overall there are 5 python scripts of interest:
    1.1. run_data_pipeline.py (Prepares the data and loads to database)
    1.2. train_drl_model.py (Trains DRL-Trader)
    1.3. train_neat_model.py (Trains NEAT-Trader)
    1.4. perform_benchmarking.py (Computes Benchmarking metrics)
    1.5. tables_and_figures.py (Jupyter Notebook with tables and figures)

2. The scripts that produce results
- In general the script run_data_pipeline.py must be ran first to setup the database. However
 it will already have been run upon providing the code. 
- As the scripts for training DRL-Trader and NEAT-Trader are running multiple days on standard
 hardware, they do not need to be ran again. The files produces by train_drl_model.py are in the
 following directory: src/Benchmarking/trained_benchmark_models. The files produces by 
 train_neat_model.py are in this directory: src/Output
- The perform_benchmarking.py script has been ran as well. Its output are the following two files:
 comparison_frame_test_set_.csv and comparison_frame_train_set_.csv
- The Jupyter Notebook tables_and_figures can be used to reproduce tables and figures presented in the 
 thesis

3. How to replicate results
- All scripts have a broad range of external dependencies. For convenience and to prevent any system
 incompatibilities multiple docker images have been produced that allow to run scripts in an isolated
 container that will automatically install all required external dependencies. There is no need to install
 python manually on the host machine. 
- Overall there are 3 docker images:
  3.1 run_all (This image runs all 5 python scripts and might ran multiple days)
  3.2 run_benchmarking (This image does not re-train the models. It relies on the already trained models,
  but re-runs all back-testing and benchmarking producing the respective .csv-files mentioned in item 2. )
  3.3 figures_tables (This docker image runs a container from which the Jupyter Notebook is hosted.)

4. How to run docker containers run_all and run_benchmarking
Docker Desktop needs to be installed and (!) running. Get it here: https://docs.docker.com/desktop/

Step 1: To run the respective docker container the images must be loaded, navigate to the NeatTrader
directory using the CLI and run the following command

docker load -i <image_name>.tar (replace <image_name> by run_all or run_benchmarking)

Step 2: To run the loaded docker container run the following command:

docker run -v /path/on/host:/app <image_name> (replace <image_name> by run_all or run_benchmarking)


Step 3: The docker images run_all and run_benchmarking save their respective results within the container and not
on the host machine, which is why they need to be copied to the host upon completion.
To copy the created files to you local machine run the following command:

docker create --name dummy temporary_dummy
docker cp temporary_dummy:/app /path/on/host (replace /path/on/host by the directory you want to save the results in)

5. How to run docker the docker container figures_tables
Docker Desktop needs to be installed and (!) running. Get it here: https://docs.docker.com/desktop/

Step 1: To run the docker container for the notebook the image must be loaded. Navigate to the NeatTrader
directory using the CLI and run the following command:

docker load -i <figures_tables>.tar 

Step 2: Run the following command to run the docker container:

docker run -it -p 8888:8888 <figures_tables>

Step 3: A link will appear in the console. Click on it, to open the jupyter notebook in your browser. You will see 
the top-level directory. Click on the tables_and_figures.ipynb files to open and run the notebook. 