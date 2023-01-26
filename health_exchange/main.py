import json
import warnings
import os
from argparse import Namespace
from typing import Dict, Generator, List
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import re
import numpy as np
import pandas as pd
import mlflow
import csv
import time
import typer
from config.config import logger
from health_exchange.utils import utils
from health_exchange import elastic, sent_transformer, evaluate, data
import torch
import pickle

# globals
data_dir = "./data"
posts_filepath = os.path.join(data_dir, "search_corpus.csv")
test_queries_filepath = os.path.join(data_dir, "test_queries.txt")
repo_dir = "./artifacts_repo"
tokenizer = RegexpTokenizer(r'\w+')
stopwords = stopwords.words("english")
stopword_pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
porter = PorterStemmer()

warnings.filterwarnings("ignore")


# initialize typer CLI app
app = typer.Typer()


def preprocess_doc(text: str, max_words: int, lower=True, remove_punct=True, remove_stops=True, stem=True) -> str:
    """Preprocesses a document/snippet by applying a selection of text operations, as defined in the model's
    configuration.


    Args:
        text (str): the text (document, snippet) to process
        max_words (int): the ideal number of words in text
        lower (bool, optional): Whether to lower-case the text. Defaults to True.
        remove_punct (bool, optional): Whether to remove punctuation. Defaults to True.
        remove_stops (bool, optional): Whether to remove skip-words. Defaults to True.
        stem (bool, optional): Whether to stem the words to their lemma (=root) form. Defaults to True.

    Returns:
        str: _description_
    """
    text = text.strip()
    text = (" ".join(text.split()[:max_words]) + ' ...' if len(text.split()) > max_words+1 else text)
    # strip punctuation
    if remove_punct:
        text = " ".join(tokenizer.tokenize(text))
    if lower:
        text = text.lower()
    if remove_stops:
        text = stopword_pattern.sub("", text)
    if stem:
        text = " ".join(map(porter.stem, text.split()))
    return text

def preprocess_corpus(args: Namespace, corpus_file: str, columns: List[str], prefix_column: str) -> Generator:
    """Top-level function to preprocess a set of documents/snippets (=corpus)

    Args:
        args (Namespace): the model's configuration
        corpus_file (str): csv file where rows list id, title, body, tags of all the documents/snippets in the search corpus
        columns (List[str]): the columns to preprocess (cf. 'preprocess_doc')
        prefix_column (str): column to use as a prefix to the document/snippet body

    Raises:
        NameError: _description_

    Yields:
        Generator: _description_
    """
    with open(corpus_file, 'r', encoding='utf-8') as fin:
        reader = csv.DictReader(fin)
        header = reader.fieldnames
        if prefix_column is not None and not (prefix_column in header):
            raise NameError(f"Error: prefix_columns not found in csv file")
        for row in reader:  
            if prefix_column == 'Tags':
                prefix = utils.textify_tags(row['Tags'])
                out = prefix + " " + " ".join([preprocess_doc(row[column], max_words=args.prep_max_words_per_snippet, 
                                                lower=args.prep_lower, remove_punct=args.prep_remove_punct, 
                                                remove_stops=args.prep_skipw, stem=args.prep_stem) for column in columns])
            elif len(prefix_column) > 0:
                logger.error("preprocess_corpus: option prefix_column other than Tags not implemented yet")
            else:
                out = " ".join([preprocess_doc(row[column], max_words=args.prep_max_words_per_snippet, 
                                                lower=args.prep_lower, remove_punct=args.prep_remove_punct, 
                                                remove_stops=args.prep_skipw, stem=args.prep_stem) for column in columns])
            yield (out, row['Id'], row['SnipIdx'], row['Snippet'], row['Tags'])

@app.command()
def prepare_data():
    """Extracts text from posts.xml, breaks up Body text into snippets, filters out data rows with missing fields, 
    shortens data file to 20K rows for quicker processing. 

    Args:
        args_file (str): the model's configuration (Elasticsearch or Sentence Transfomer)
    """
    # args = Namespace(**utils.load_dict(filepath=args_file))
    data_dir = './data'
    health_data_path = os.path.join(os.path.abspath(data_dir), "health.stackexchange.com")
    posts_xml_file = os.path.join(health_data_path, "Posts.xml")
    posts_fields = ['Id', 'PostTypeId', 'CreationDate', 'Score', 'ViewCount', 'Body', 
                'OwnerUserId', 'LastEditDate', 'LastActivityDate', 'Title',
                'Tags', 'AnswerCount', 'CommentCount', 'ContentLicense']
    # read Posts.xml into dataframe and csv
    posts_df = data.extract_raw_data(posts_xml_file, posts_fields)
    print(f"Extracted {len(posts_df)} raw posts from {os.path.basename(posts_xml_file)}")
    posts_csv_file = os.path.join(data_dir, "posts.csv")
    posts_df.to_csv(posts_csv_file, encoding='utf-8', index=False, quoting=csv.QUOTE_NONNUMERIC)
    # remove rows with empty Body, Title, and/or Tags
    validated_posts_csv_file = os.path.join(data_dir, "posts_validated.csv")
    data.update_csv(posts_csv_file, validated_posts_csv_file, ['Body', 'Title', 'Tags'], posts_fields, data.remove_empty_cols_rows)
    validated_posts_df = pd.read_csv(validated_posts_csv_file, sep=',', usecols=['Id','Title','Body','Tags','ViewCount'])
    # preprocess dataframe
    snippets_csv_file = os.path.join(data_dir, "posts_snippets_validated.csv")
    wcs = data.preprocess_to_snippets(validated_posts_df, max_words=30, min_words=5, outfile=snippets_csv_file)
    print(f"Snippet Word Counts: Mean={round(np.mean(wcs),3)}, Median= {np.median(wcs)} STD={round(np.std(wcs), 3)}")
    # select top 20K rows by ViewCount and write to file
    data.select_top_rows_by_viewcount(validated_posts_df, snippets_csv_file)



@app.command()
def create_index(args_file: str):
    """Creates a search index, given configuration arguments.

    Args:
        args_file (str): the model's configuration (Elasticsearch or Sentence Transfomer)

    Raises:
        ValueError: Only 'elasticsearch' and 'sent_transformer' model types are supported
    """
    args = Namespace(**utils.load_dict(filepath=args_file))
    repo_dir = "./artifacts_repo"
    # elasticsearch
    if args.model_type == "elasticsearch":
        repo_dir = os.path.join(repo_dir, "elastic")
        index_name = "healthex_" + os.path.splitext(os.path.basename(args_file))[0][5:]  # = healthex_elastic_1
        mlflow.set_experiment(experiment_name=index_name)
        with mlflow.start_run(run_name=args.run_name + "_create-index"):
            run_id = mlflow.active_run().info.run_id
            logger.info(f"Run id: {run_id}")
            # connect to elasticsearch server
            es = elastic.connect_to_es()
            # create index
            elastic.create_index(es, index_name=index_name)
            # store documents and setup document info
            prepped_snippet_to_display_info = dict()
            start = time.time()
            records = []
            for text_info in preprocess_corpus(args, posts_filepath, columns=[args.prep_col], prefix_column=args.prep_prefix_col):
                record = {'snippet': text_info[0]}
                records.append(json.dumps(record))
                prepped_snippet_to_display_info[text_info[0]] = text_info[1:]
            elastic.bulk_store_records(es, records, index_name)
            logger.info("... create_index: Elasticsearch Index completed in " + time.strftime("%H:%M:%S", time.gmtime(time.time() - start)))
            # save snippet info to dir "artifacts_repo" -- to display legible search results
            snippet_info_filepath = os.path.join(repo_dir, index_name + "_info.json")
            utils.write_dict(prepped_snippet_to_display_info, snippet_info_filepath, 
                            f"create_index: Written Elasticsearch document info file to: {snippet_info_filepath}",
                            f"create_index: Could not write Elasticsearch document info file to {snippet_info_filepath};")
            mlflow.log_params(vars(args))
        mlflow.end_run()
    # sentence transformer
    elif args.model_type == "sent_transformer":
        repo_dir = os.path.join(repo_dir, "sent_trans")
        index_name = "healthex_" + os.path.splitext(os.path.basename(args_file))[0][5:]  # = healthex_senttrans_1
        mlflow.set_experiment(experiment_name=index_name)
        with mlflow.start_run(run_name=args.run_name + "_create-index"):
            run_id = mlflow.active_run().info.run_id
            logger.info(f"Run id: {run_id}")
            device, dev_name = ('cuda:0', 'GPU') if torch.cuda.is_available() else ('cpu', 'CPU')
            # load model
            embedder = sent_transformer.load_model(args.model_filepath, device=device)
            # setup document info, to display search results
            prepped_snippet_to_display_info = dict()
            # encode corpus as list of embeddings
            docs = []
            for text_info in preprocess_corpus(args, posts_filepath, columns=[args.prep_col], prefix_column=args.prep_prefix_col):
                docs.append(text_info[0])
                prepped_snippet_to_display_info[text_info[0]] = text_info[1:]
            #save docs to file, for display during search
            corpus_filepath = os.path.join(repo_dir, index_name + "_corpus.pkl")
            utils.pickle_list_to_file(docs, corpus_filepath, f"create_index: Written search corpus to file {corpus_filepath}",
                                                             f"create_index: Could not write search corpus file {corpus_filepath}")
            # encode corpus into embeddings
            start = time.time()
            corpus_embeddings = sent_transformer.encode_corpus(embedder=embedder, corpus=docs)
            logger.info("create_index: Embeddings created for corpus documents in " + time.strftime("%H:%M:%S", time.gmtime(time.time() - start)))
            assert len(docs) == len(corpus_embeddings)  # snippets and their encoding must match
            # store embeddings (=search index) to file in repo_dir folder
            index_filepath = os.path.join(repo_dir, index_name + "_index.pkl")
            utils.pickle_list_to_file(corpus_embeddings, index_filepath, f"create_index: Written corpus embeddings to {index_filepath}",
                                                                         f"create_index: Could not write corpus embeddings to {index_filepath}")
            # save snippet info to dir "artifacts_repo" -- to display legible search results
            snippet_info_filepath = os.path.join(repo_dir, index_name + "_info.json")
            utils.write_dict(prepped_snippet_to_display_info, snippet_info_filepath, f"create_index: Written snippet info file to {snippet_info_filepath}",
                                                                                     f"create_index: Could not write snippet infor file to {snippet_info_filepath}")
            mlflow.log_params(vars(args))
            mlflow.log_artifacts(repo_dir)
        mlflow.end_run()
    # unsupported framework
    else:
        raise ValueError("create_index: framework not supported")


@app.command()
def delete_index(args_file: str):
    """Deletes a search index, if such exists, given configuration arguments.

    Args:
        args_file (str): the model's configuration (Elasticsearch or Sentence Transfomer)

    Raises:
        ValueError: Only 'elasticsearch' and 'sent_transformer' model types are supported
    """
    args = Namespace(**utils.load_dict(filepath=args_file))
    # elasticsearch
    if args.model_type == "elasticsearch":
        es = elastic.connect_to_es()
        index_name = "healthex_" + os.path.splitext(os.path.basename(args_file))[0][5:]  # = healthex_elastic_1
        es.indices.delete(index=index_name, ignore=[400, 404])
        logger.info("delete_index: index " + index_name + " has been deleted")
    # sentence transformer
    elif args.model_type == "sent_transformer":
        repo_dir = os.path.join(repo_dir, "sent_trans")
        index_name = "healthex_" + os.path.splitext(args_file)[0][12:]  # = healthex_senttrans_1
        index_filepath = os.path.join(repo_dir, index_name + "_index.pkl")
        utils.delete_file(index_filepath, "delete_index: index file " + index_filepath + " has been removed", 
                                          "delete_index: index_file " + index_filepath + " could not be deleted;")
        snippet_info_filepath = os.path.join(repo_dir, index_name + "_info.json")
        utils.delete_file(snippet_info_filepath, "delete_index: snippet info file " + snippet_info_filepath + " has been removed", 
                                                 "delete_index: snippet info file " + snippet_info_filepath + " could not be deleted;")
        corpus_filepath = os.path.join(repo_dir, index_name + "_corpus.pkl")
        utils.delete_file(corpus_filepath, "delete_index: corpus file " + corpus_filepath + " has been removed", 
                                           "delete_index: corpus file " + corpus_filepath + " could not be deleted")
    else:
        raise ValueError("delete_index: framework not supported")


@app.command()
def run_test_queries(args_file: str):
    """Runs the model over a set of test queries and collects the results in a file, given configuration arguments.

    Args:
        args_file (str): the model's configuration (Elasticsearch or Sentence Transfomer)

    Raises:
        ValueError: Only 'elasticsearch' and 'sent_transformer' model types are supported
    """
    args = Namespace(**utils.load_dict(filepath=args_file))
    test_queries = utils.read_text_file(test_queries_filepath)
    repo_dir = "./artifacts_repo"
    # elasticsearch
    if args.model_type == "elasticsearch":
        es = elastic.connect_to_es()
        repo_dir = os.path.join(repo_dir, "elastic")
        index_name = "healthex_" + os.path.splitext(os.path.basename(args_file))[0][5:]  # = healthex_elastic_1
        mlflow.set_experiment(experiment_name=index_name )
        with mlflow.start_run(run_name=args.run_name+ "_run_queries", nested=False):
            run_id = mlflow.active_run().info.run_id
            logger.info(f"Run id: {run_id}")
            snippet_info_filepath = os.path.join(repo_dir, index_name + "_info.json")
            results_file = os.path.join(repo_dir, index_name + "_results.csv")
            # write search results to file
            with open (results_file, 'w', encoding='utf-8') as results:
                columns = ['query','grade','rank','id','snippet','tags','notes']
                writer = csv.DictWriter(results, fieldnames=columns)
                writer.writeheader()
                for query_raw in test_queries:
                    prepped_query = preprocess_doc(query_raw, max_words=args.prep_max_words_per_snippet, 
                                        lower=args.prep_lower, remove_punct=args.prep_remove_punct, 
                                        remove_stops=args.prep_skipw, stem=args.prep_stem)
                    results = elastic.get_query_results(es, query_raw, prepped_query, index_name, snippet_info_filepath)
                    for result in results:
                        writer.writerow(result)
            mlflow.log_artifacts(repo_dir)
        mlflow.end_run()
    # sentence transformer
    elif args.model_type == "sent_transformer":
        repo_dir = os.path.join(repo_dir, "sent_trans")
        index_name = "healthex_" + os.path.splitext(os.path.basename(args_file))[0][5:]  # = healthex_senttrans_1
        mlflow.set_experiment(experiment_name=index_name)
        with mlflow.start_run(run_name=args.run_name+ "_run_queries"):
            run_id = mlflow.active_run().info.run_id
            logger.info(f"Run id: {run_id}")
            # load search result info to display in search results
            snippet_info_filepath = os.path.join(repo_dir, index_name + "_info.json")
            if not os.path.exists(snippet_info_filepath):
                raise ValueError("rn_test_queries: could not find Snippet Info file")
            snippet_info = utils.load_dict(snippet_info_filepath)
            # read embeddings index from saved file
            index_filepath = os.path.join(repo_dir, index_name + "_index.pkl")
            corpus_embeddings = utils.read_pickled_file(index_filepath, f"run_test_queries: loaded embeddings from {index_filepath}", 
                                                                        f"run_test_queries: could not read embeddings from {index_filepath}")
            # read persisted corpus file
            corpus_filepath = os.path.join(repo_dir, index_name + "_corpus.pkl")
            corpus = utils.read_pickled_file(corpus_filepath, f"run_test_queries: loaded corpus file from {corpus_filepath}",
                                                              f"run_test_queries: coult not read corpus from {corpus_filepath}")
            # write search results to file
            device, dev_name = ('cuda:0', 'GPU') if torch.cuda.is_available() else ('cpu', 'CPU')
            # load model
            embedder = sent_transformer.load_model(args.model_filepath, device=device)
            results_file = os.path.join(repo_dir, index_name + "_results.csv")
            with open (results_file, 'w', encoding='utf-8') as results:
                columns = ['query','grade','rank','id','snippet','tags','notes']
                writer = csv.DictWriter(results, fieldnames=columns)
                writer.writeheader()
                for query in test_queries:
                    results = sent_transformer.get_search_results_for_query(query, embedder, corpus_embeddings, corpus, snippet_info, args.num_search_results)
                    for result in results:
                        writer.writerow(result)
            mlflow.log_artifacts(repo_dir)
        mlflow.end_run()
    # unsupported framework
    else:
        raise ValueError("run_test_queries: framework not supported")


@app.command()
def evaluate_run(args_file: str):
    """Produces search metrics for the test queries, based on a human-graded "true"
    results file, given configuration arguments.

    Args:
        args_file (str): the model's configuration (Elasticsearch or Sentence Transfomer)

    Raises:
        ValueError: Only 'elasticsearch' and 'sent_transformer' model types are supported
    """
    args = Namespace(**utils.load_dict(filepath=args_file))
    repo_dir = "./artifacts_repo"
    results_true_file = os.path.join(data_dir, "health_search_results_true.csv")
    if args.model_type == "elasticsearch" or args.model_type == "sent_transformer":
        index_name = "healthex_" + os.path.splitext(os.path.basename(args_file))[0][5:]
        mlflow.set_experiment(experiment_name=index_name)
        with mlflow.start_run(run_name=args.run_name + "_evaluation_metrics", nested=False):
            run_id = mlflow.active_run().info.run_id
            logger.info(f"Run id: {run_id}")
            if args.model_type == "elasticsearch": repo_dir = os.path.join(repo_dir, "elastic") 
            else: repo_dir = os.path.join(repo_dir, "sent_trans")
            model_results_file = os.path.join(repo_dir, index_name + "_results.csv")
            metrics = evaluate.get_metrics(results_true_file=results_true_file, model_results_file=model_results_file)
            mlflow.log_metrics(metrics)
            print(json.dumps(metrics, indent=4))
        mlflow.end_run()
    else:
        raise ValueError("evaluate_run: framework not supported")


def print_interactive_usage():
    print("Usage: enter search query after prompt; enter 'q' to exit\n")


@app.command()
def get_search_results_interactive(args_file: str):
    """Command line interface to ask new questions and see the results provided 
    by the experiment's configuration

    Args:
        args_file (str): the model's configuration (Elasticsearch or Sentence Transfomer)

    Raises:
        ValueError: Only 'elasticsearch' and 'sent_transformer' model types are supported
    """
    args = Namespace(**utils.load_dict(filepath=args_file))
    repo_dir = "./artifacts_repo"
    # elasticsearch
    if args.model_type == "elasticsearch":
        print_interactive_usage()
        es = elastic.connect_to_es()
        index_name = "healthex_" + os.path.splitext(os.path.basename(args_file))[0][5:]
        repo_dir = os.path.join(repo_dir, "elastic")
        snippet_info_filepath = os.path.join(repo_dir, index_name + "_info.json")
        query = ""
        while True:
            query = input("Query: ")
            if query == 'q': exit()
            prepped_query = preprocess_doc(query, max_words=args.prep_max_words_per_snippet, 
                                lower=args.prep_lower, remove_punct=args.prep_remove_punct, 
                                remove_stops=args.prep_skipw, stem=args.prep_stem)
            results = elastic.get_query_results(es, query, prepped_query, index_name, snippet_info_filepath)
            for result in results:
                print(f"   {result['rank']}  [id: {result['id']} tags: {utils.textify_tags(result['tags'])}]  {result['snippet']}")
    # Sentence Transformer
    elif args.model_type == "sent_transformer":
        print_interactive_usage()
        index_name = "healthex_" + os.path.splitext(os.path.basename(args_file))[0][5:]
        repo_dir = os.path.join(repo_dir, "sent_trans")
        snippet_info_filepath = os.path.join(repo_dir, index_name + "_info.json")
        if not os.path.exists(snippet_info_filepath):
            raise ValueError("rn_test_queries: could not find Snippet Info file")
        snippet_info = utils.load_dict(snippet_info_filepath)
        # read embeddings index from saved file
        index_filepath = os.path.join(repo_dir, index_name + "_index.pkl")
        corpus_embeddings = utils.read_pickled_file(index_filepath, f"run_test_queries: loaded embeddings from {index_filepath}", 
                                                                    f"run_test_queries: could not read embeddings from {index_filepath}")
        # read persisted corpus file
        corpus_filepath = os.path.join(repo_dir, index_name + "_corpus.pkl")
        corpus = utils.read_pickled_file(corpus_filepath, f"run_test_queries: loaded corpus file from {corpus_filepath}",
                                                            f"run_test_queries: coult not read corpus from {corpus_filepath}")
        device, dev_name = ('cuda:0', 'GPU') if torch.cuda.is_available() else ('cpu', 'CPU')
        # load model
        embedder = sent_transformer.load_model(args.model_filepath, device=device)
        query = ""
        while True:
            query = input("Query: ")
            if query == 'q': exit()
            results = sent_transformer.get_search_results_for_query(query, embedder, corpus_embeddings, corpus, snippet_info, args.num_search_results)
            for result in results:
                print(f"   {result['rank']}  [id: {result['id']} tags: {utils.textify_tags(result['tags'])}]  {result['snippet']}  ({result['notes']})")                  
    else:
        raise ValueError("evaluate_run: framework not supported")


@app.command()
def encode_corpus_cpu_only(args_file: str):
    """Recreates sentence embeddings for the snippets in the search corpus for
    systems with cpu-only. This will overwrite the *_index.pkl file in artifacts_repo/sent_trans
    that comes with the cloned repository. The latter was created for GPU enabled systems.
    ** NOTE **: do not run this command if you have an available CUDA (GPU) device!

    Args:
        args_file (str): the model's configuration (Sentence Transformer)
    """
    args = Namespace(**utils.load_dict(filepath=args_file))
    repo_dir = "./artifacts_repo"
    repo_dir = os.path.join(repo_dir, "sent_trans")
    index_name = "healthex_" + os.path.splitext(os.path.basename(args_file))[0][5:]
    corpus_filepath = os.path.join(repo_dir, index_name + "_corpus.pkl")
    with open(corpus_filepath, 'rb') as fin:
        docs = pickle.load(fin)
    device, dev_name = ('cuda:0', 'GPU') if torch.cuda.is_available() else ('cpu', 'CPU')
    embedder = sent_transformer.load_model(args.model_filepath, device=device)
    corpus_embeddings = sent_transformer.encode_corpus(embedder=embedder, corpus=docs)
    index_filepath = os.path.join(repo_dir, index_name + "_index.pkl")
    utils.pickle_list_to_file(corpus_embeddings, index_filepath, f"create_index: Written corpus embeddings to {index_filepath}",
                                                                 f"create_index: Could not write corpus embeddings to {index_filepath}")
    

if __name__ == "__main__":
    app()
