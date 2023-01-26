<pre>
## Install Module
HealthSearch> python3 -m pip install -e .

## Run tests
- ### All tests:
HealthSearch> python3 -m pytest 
- ### All tests in a file:
HealthSearch> python3 -m pytest tests/code/test_data.py
- ### One test
HealthSearch> python3 -m pytest tests/code/test_data.py::test_get_word_len
## Tests Coverage
HealthSearch> python3 -m pytest --cov health_exchange --cov-report html

## health_exchange CLI
# -- general help
HealthSearch> python3 health_exchange/main.py --help
    Usage: main.py [OPTIONS] COMMAND [ARGS]...
    Commands:
    encode-corpus-cpu-only          Recreates the index (corpus embeddings) for CPU-only systems
    create-index                    Creates a search index, given configuration arguments
    delete-index                    Deletes a search index, if such exists
    evaluate-run                    Produces search metrics for the test queries
    prepare-data                    Perfoms ETL to create search corpus
    run-test-queries                Runs the model over a set of test queries
    get-search-results-interactive  Get top 5 search results for new user queries
# -- command help
python3 health_exchange/main.py create-index --help

# for CPU-only systems -- DO NOT RUN if you have a CUDA enabled GPU!
python3 health_exchange/main.py encode-corpus-cpu-only "config/args_senttrans_1.json"
# -- execute command to run experiements
python3 health_exchange/main.py prepare-data
python3 health_exchange/main.py create-index "config/args_elastic_1.json" 
python3 health_exchange/main.py create-index "config/args_elastic_1.1.json" 
python3 health_exchange/main.py create-index "config/args_senttrans_1.json" 
python3 health_exchange/main.py delete-index "config/args_elastic_1.json"
python3 health_exchange/main.py delete-index "config/args_elastic_1.1.json"
python3 health_exchange/main.py run-test-queries "config/args_elastic_1.json"  
python3 health_exchange/main.py run-test-queries "config/args_elastic_1.1.json"
python3 health_exchange/main.py run-test-queries "config/args_senttrans_1.json" 
python3 health_exchange/main.py evaluate-run "config/args_elastic_1.json"       
python3 health_exchange/main.py evaluate-run "config/args_elastic_1.1.json"   
python3 health_exchange/main.py evaluate-run "config/args_senttrans_1.json"
# -- ask queries interactively
python3 health_exchange/main.py get-search-results-interactive "./config/args_elastic_1.json"
python3 health_exchange/main.py get-search-results-interactive "./config/args_senttrans_1.json"


## SHELL 
## runs all phases of an experiment: create-index, run-test-queries, and evaluate-run
HealthSearch> ./run-experiment.sh config/args_elastic_1.json
HealthSearch> ./run-experiment.sh config/args_elastic_1.1.json
HealthSearch> ./run-experiment.sh config/args_senttrans_1.json

## Elastisearch
## List current indices
curl -XGET localhost:9200/_cat/indices

## REFERENCE STUFF
## MLFLOW Experiment Tracking
## start server
mlflow server -h 0.0.0.0 -p 8000 --backend-store-uri ./experiments &
    [2023-01-03 13:34:40 +0000] [5161] [INFO] Starting gunicorn 20.1.0
    [2023-01-03 13:34:40 +0000] [5161] [INFO] Listening at: http://0.0.0.0:8000 (5161)
## MLFLOW CLI
# delete a run by its id (find id on the server, by clicking the run name)
mlflow runs delete --run-id <RUN_ID>


## Data Validation with great_expectations: in a new environment, you might want to run 
the following commands to use great_expectations:
cd tests
great_expectations init
great_expectations datasource new 
### ...complete Notebook 
great_expectations suite new
### ...run 'great_expectations suite edit all_posts' to re-run tests after making changes
## Create checkpoint for later re-validation
great_expectations checkpoint new <checkpoint_name>

</pre>