<h1>HealthSearch: A comparison of ElasticSearch vs. Semantic Search</h1>
<p>This project features a comparison of two search frameworks: ElasticSearch and Semantic Search, with a discussion of their pros and cons (cf. Notebooks/Models.ipynb). </p>
<p>It was inspired by "MadeWithML" by Goku Mohandas [2022] (https://madewithml.com/).</p>
<p>Both frameworks are implemented with standard Python libraries. Semantic Search is implemented using a pre-trained (and fine-tuned) Transformer model, 
'all-MiniLM-L12-v2', which performs well out-of-the-box (cf. https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2).</p> 
<p>The search data is the Health dataset from StackExchange (https://health.stackexchange.com) </p>
<p>Start with the Notebooks. They contain the main narrative for this project.</p>
<h2>Contents</h2>
<ul><li>Notebooks: Models.ipynb gives the main narrative of the project; Data.ipynb contains data analysis. </li>
    <li>Models: the 'all-MiniLM-L12-v2' model</li>
    <li>config: configuration for the Python module and for the experiments</li>
    <li>health_exchange: the Python module</li>
    <li>data: health dataset from StackExchange, graded test set</li>
    <li>experiments: MLFlow experiments data</li>
    <li>artifacts_repo: experiments' artifacts repository</li>
    <li>tests: pytest unit-tests. <i>TO-DO: write lots more tests!</i></li>
    <li>COMMANDS.md: list of helpful CLI operations for developing the project</li>
    <li>requirements.txt: required python packages</li>
</ul>
<h2>Setup</h2>
<p>After looking at the Notebooks, if you wish to play with the environment, you may set it up on your local machine as follows.</p>
<i>NOTE: Use of a GPU is highly recommended</i>
<pre>
cd ElasticSearch-vs-SemanticSearch-with-Transformers
# Recommended: create a virtual env, e.g. with miniconda
conda create -n myenv python=3.7
conda activate myenv
# get up-to-date with basic packages
pip install --upgrade pip
pip install --upgrade setuptools
# install pytorch: this depends on your local environment and machine. Follow the instructions on https://pytorch.org/get-started/locally/
# If you have an nvidia GPU and CUDA-XX.X installed, you should install torch with GPU accelleration. Otherwise, install 'cpuonly', but 
# be prepared for long waits, especially when interacting with the Transformers models.
# install the package dependencies
python3 -m pip install -r requirements.txt
# install the module
python3 -m pip install -e .
# If you have a cpu-only (no CUDA/GPU) system, you need to re-create the corpus embeddings for Semantic Search. 
# The artifacts_repo/sent_trans/healthex_senttrans_1_index.pkl that comes with the installation is for GPU-enabled systems.
## DO NOT RUN if you have a CUDA enabled GPU!
python3 health_exchange/main.py encode-corpus-cpu-only "config/args_senttrans_1.json"

</pre>
<h2>Environment Hands-on</h2>
<p>The code to interact with the enviroment is in health_exchage/main.py. There is a CLI interface to perform ETL and run 
experiments, either with Elasticsearch, or with the Transformers model. See COMMANDS.md for specific instructions.</p>
<p><b>NOTE</b>: you don't need to run any of the CLI, as this has already been done. However, if you want to experiment
with the environment, here are some ideas:
<ul>
    <li>Make changes to the ETL code (cf. data.py) and re-run 'prepare-data' to produce a different searchable corpus: change
    the size of the snippets, or the snippets' "context": e.g. replace the tags with the title(?)</li>
    <li>Play with the search frameworks by asking questions and analyzing the results. To do so, cf. '# ask queries 
    interactively' in COMMANDS.md. </li>
    <li>Run a new experiment: start by creating a new configuration 'args_<FRAMEWORK>_X.X.json' in folder config; then run 
    the various phases of the experiment as listed in COMMANDS.md.</li>
</ul>
