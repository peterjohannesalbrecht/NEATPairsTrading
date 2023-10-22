# NeuroEvolution of Augmenting Topologies for Pairs Trading
This repository consists of the Software of the master's thesis titled: 'Reinforcement Learning for Pairs Trading - An approach
using the NEAT algorithm. The steps below are a guide on how to replicate the thesis' results. 

Note, that for the repository on github, the docker images are not included in the repository. -- <br> 
They must be created by running the dockerfile from the main directory and the /dockerization -folder --


## 1. Software Architecture
- All core modules written and used for this thesis are in the `src` folder.
- All scripts that produce (intermediary) results are in the top-level directory.
- Overall there are 5 python scripts of interest:
    - 1.1. run_data_pipeline.py (Prepares the data and loads to database)
    - 1.2. train_drl_model.py (Trains DRL-Trader)
    - 1.3. train_neat_model.py (Trains NEAT-Trader)
    - 1.4. perform_benchmarking.py (Computes Benchmarking metrics)
    - 1.5. tables_and_figures.py (Jupyter Notebook with tables and figures)

## 2. The scripts that produce results
- In general the script run_data_pipeline.py must be ran first to setup the database. However<br>
 it will already have been run upon providing the code. 
- As the scripts for training DRL-Trader and NEAT-Trader are running multiple days on standard <br>
 hardware, they do not need to be ran again. The files produces by train_drl_model.py are in the<br>
 following directory: src/Benchmarking/trained_benchmark_models. The files produces by <br>
 train_neat_model.py are in this directory: src/Output
- The perform_benchmarking.py script has been ran as well. Its output are the following two files:<br>
 comparison_frame_test_set_.csv and comparison_frame_train_set_.csv
- The Jupyter Notebook tables_and_figures can be used to reproduce tables and figures presented in the <br>
 thesis

## 3. How to replicate results
- All scripts have a broad range of external dependencies. For convenience and to prevent any system<br>
 incompatibilities multiple docker images have been produced that allow to run scripts in an isolated<br>
 container that will automatically install all required external dependencies. There is no need to install<br>
 python manually on the host machine. 
- Overall there are 3 docker images:
  - 3.1 run_all (This image runs all 5 python scripts and might ran multiple days)
  - 3.2 run_benchmarking (This image does not re-train the models. It relies on the already trained models,
  - but re-runs all back-testing and benchmarking producing the respective .csv-files mentioned in item 2. )
  - 3.3 tables_figures (This docker image runs a container from which the Jupyter Notebook is hosted.)
- All docker images are located in the dockerization folder of the top-level directory
- Of course everything can be run without docker but this requires a manual installation of <br> all packages using the right version as well as installing the correct python version

## 4. How to run docker containers run_all and run_benchmarking
Docker Desktop needs to be installed and (!) running. Get it here: [Docker Desktop](https://docs.docker.com/desktop/) <br>

Step 1: To run the respective docker container the images must be loaded, navigate to the dockerization <br>
directory using the CLI and run the following command

```docker load -i <image_name>.tar ```
Please note, that loading can take several minutes because of the large file-size! 
Step 2: To run the loaded docker container run the following command: <br>

```docker run -v "$(pwd)":/app/src/Output/comparison_frame/ <image_name>```

Using volume mounting, written out files will be saved to your working directory on the host.
Please replace <image_name> with run_all or run_benchmarking. <br>
If you are NOT using a UNIX-based system, but for example windows, you might have to replace <br> \$(pwd) with \%cd\% in the Command Prompt or ${PWD} in PowerShell. 



## 5. How to run the docker container tables_figures
Docker Desktop needs to be installed and (!) running. Get it here: [Docker Desktop](https://docs.docker.com/desktop/) <br>

Step 1: To run the docker container for the notebook the image must be loaded. Navigate to the dockerization <br>
directory using the CLI and run the following command: <br>

```docker load -i tables_figures.tar```

Please note, that loading can take several minutes because of the large file-size! 
<br>

Step 2: Run the following command to run the docker container: <br>

```docker run -it -p 8888:8888 tables_figures```

Step 3: A link will appear in the console. Click on it, to open the jupyter notebook in your browser. You will see <br>
the top-level directory. Click on the tables_and_figures.ipynb files to open and run the notebook.

## 5. How to view the raw data from Refinitiv Eikon Datastream
The raw data is located in src/Data/data_PJA.xlsx.
