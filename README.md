# Optical properties of black carbon (BC) at various stages of aging: comprehensive dataset and machine learning-based prediction algorithm
This repository contains code to reproduce figures and experiments described in

Optical properties of black carbon (BC) at various stages of aging: comprehensive dataset and machine learning-based prediction algorithm
Jaikrishna Patil, Baseerat Romshoo, Tobias Michels, Thomas Müller, Marius Kloft, and Mira Pöhlker
(TODO: title might change, add link to paper)


## Installing required software
Running the experiments and notebook in this repository requires a working Python interpreter with several packages installed. We recommend using [conda](https://conda.io/projects/conda/en/latest/index.html) to setup a virtual environment:
1. Follow the instructions at the [conda website](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) to download and install Miniconda (or Anaconda if you prefer) for your OS.
3. Clone this repository: (TODO: link might change)
   ```commandline
   tobias@tobias-Laptop:~$ git clone https://github.com/tmichels-tukl/Optical-properties-of-black-carbon-aggregates.git bcfa_experiments
   Cloning into 'bcfa_experiments'...
   remote: Enumerating objects: 50, done.
   remote: Counting objects: 100% (11/11), done.
   remote: Compressing objects: 100% (10/10), done.
   remote: Total 50 (delta 3), reused 7 (delta 1), pack-reused 39
   Receiving objects: 100% (50/50), 43.13 MiB | 9.20 MiB/s, done.
   Resolving deltas: 100% (11/11), done.
   tobias@tobias-Laptop:~$ cd bcfa_experiments
   tobias@tobias-Laptop:~/bcfa_experiments$ 
   ```
2. Type the following to create a new virtual environment containing the required packages to run the prediction script:
   ```commandline
   tobias@tobias-Laptop:~/bcfa_experiments$ conda env create -f conda_env.yml
   ```
   If you want to use your NVIDIA GPU to accelerate training using the Neural Network, please replace `conda_env.yml` with `conda_env_gpu.yml` in the above command.
3. To check whether the installation was successful, try running the following commands:
   ```commandline
   tobias@tobias-Laptop:~/bcfa_experiments$ conda activate BCA_exp
   (BCA_exp) tobias@tobias-Laptop:~/bcfa_experiments$ python
   Python 3.9.5 (default, Jun  4 2021, 12:28:51) 
   [GCC 7.5.0] :: Anaconda, Inc. on linux
   Type "help", "copyright", "credits" or "license" for more information.
   >>> import keras
   >>> quit()
   (BCA_exp) tobias@tobias-Laptop:~/bcfa_experiments$
   ```

## Running experiments
We use [sacred](https://sacred.readthedocs.io/en/stable/) to keep track of experiments. It allows us to change configuration parameters for experiments on the command and logs results either in a database or on disk.

To run an experiment, first setup omniboard (see the next sections) and then run it like you would any other python script:
```commandline
tobias@tobias-Laptop:~/bcfa_experiments$ conda activate BCA_exp
(BCA_exp) tobias@tobias-Laptop:~/bcfa_experiments$ cd experiments
(BCA_exp) tobias@tobias-Laptop:~/bcfa_experiments/experiments$ python random_split.py
```

You can change any configuration parameter via the command-line interface:
```commandline
(BCA_exp) tobias@tobias-Laptop:~/bcfa_experiments/experiments$ python random_split.py with "params.batch_size=64"
```
This will run the neural network training with a batch size of 64 instead of the default 32.

## Reproducing figures
The jupyter notebook `notebooks/paper_plots.ipynb` contains the code to create all figure in the main part and appendix of our paper. To run it, type the following in a console window:
```commandline
tobias@tobias-Laptop:~/bcfa_experiments$ conda activate BCA_exp
(BCA_exp) tobias@tobias-Laptop:~/bcfa_experiments$ cd notebooks
(BCA_exp) tobias@tobias-Laptop:~/bcfa_experiments/notebooks$ jupyter notebook
```
This will open a new browser tab displaying the contents of the `notebooks` directory. Click on `paper_plots.ipynb` to open the notebook. To run the entire notebook in sequence and produce all the figures select `Kernel->Restart & Run All` from the menu.

## Running omniboard
We use [omniboard](https://github.com/vivekratnavel/omniboard) to keep track of experiments. It can be run either in a local installation or using docker.

### Using Docker
The follwoing instructions should work on all platforms that support docker.

1. Install [docker](https://docs.docker.com/get-docker/) and [docker-compose](https://docs.docker.com/compose/install/).
2. `cd omniboard`
3. `docker-compose build`
4. `docker-compose up -d`
5. If you add runs for a new experiment, you need to restart Omniboard and the config generator for them to show up: `docker-compose restart config_generator omniboard`
6. To stop the services from running type `docker-compose down`.

The `docker-compose.yml` also contains an instance of [MongoExpress](https://github.com/mongo-express/mongo-express) that can be reached by opening http://localhost:8081/. This is useful for debugging problems with MongoDB.
It also exposes the MongoDB instance itself on port 27017, so CLI tools can also be used from the host.

### Without using docker
The following instructions should work on all linux distributions. Windows should be similar, but it has not been tested yet.

1. Install [MongoDB](https://docs.mongodb.com/manual/installation/) and [OmniBoard](https://vivekratnavel.github.io/omniboard/#/quick-start). 
2. Create and empty directory `mkdir -p ~/mongo/data/db`.
3. Start a mongodb instance locally `sudo mongod --dbpath ~/mongo/data/db`.
4. Run the config generation script `python scripts/generate_mongo_config_file.py`. This will generate a file called `db_config.json`. You can also specify a different name by using the `--out_file` option of the script.
5. Start the Omniboard session `OMNIBOARD_CONFIG=db_config.json omniboard`. The environment variable `OMNIBOARD_CONFIG` should point to the config file generated in the previous step.
6. As with the docker installation you need to re-run the config generator (step 4) and restart omniboard (step 5) if you add a new experiment. 

Finally, you can open http://localhost:9000/ in your browser to access omniboard.

