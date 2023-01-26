from typing import Dict, List
import csv
import os
from math import log2
import numpy as np
from config.config import logger


VERBOSE = True

def read_true_results(results_true_file: str) -> Dict:
    """Read csv file with true results and compile into a dictionary with queries as keys and 
       values as dictionaries with snippet ids as keys and grades as values.
    

    Args:
        results_true_file (str): the name of the manually graded true results file

    Returns:
        Dict: dictionary query -> dictionary snippet-id -> grade
    """
    with open(results_true_file, 'r', encoding='utf-8') as y_true:
        columns = ['query','grade','id','snippet','notes']
        reader = csv.DictReader(y_true, fieldnames=columns)
        next(y_true)  # skip header
        true_results = {}
        for row in reader:
            query = row['query']
            if query not in true_results:
                true_results[query] = {row['id']: row['grade']}
            else:
                true_results[query][row['id']] = row['grade']
    return true_results


def get_metrics(results_true_file: str, model_results_file: str) -> Dict:
    """Reads the 'true results' file and the file with search results from the model
    run, and computes relevance table and average dcg, plus some other stats

    Args:
        results_true_file (str): the file with all search results, graded by human 
        model_results_file (str): file with search results from an experiment

    Returns:
        Dict: dictionary with metrics
    """
    grade_counts = {'3': 0, '2': 0, '1': 0}
    dcg_grades = []
    metrics = {}
    true_results = read_true_results(results_true_file)
    with open(model_results_file, 'r', encoding='utf-8') as y_pred:
        columns = ['query','grade','rank','id','snippet','tags','notes']
        reader = csv.DictReader(y_pred, fieldnames=columns)
        next(y_pred)
        dcg_scores = []
        ungraded_cnt = 0
        ungraded_results = []
        result_cnt = 0
        queries = []
        for row in reader:
            rank = int(row['rank'])
            if rank > 5:
                continue
            result_cnt += 1
            if row['query'] in true_results.keys():
                query_grades = true_results[row['query']]
                if row['id'] in query_grades.keys():
                    if row['query'] not in queries: queries.append(row['query'])
                    # fill relevance table
                    grade = query_grades[row['id']]
                    if grade == '3': grade_counts['3'] += 1
                    elif grade == '2': grade_counts['2'] += 1
                    elif grade == '1': grade_counts['1'] += 1
                    else:
                        logger.error(f"Invalid or missing grade: {grade} in {results_true_file}. Please correct.")
                    # compute DCG
                    grade = int(grade)
                    dcg = ((2**grade) - 1) / log2(rank + 1)
                    dcg_scores.append(dcg)
                else:
                    logger.warning(f"Result Id: {row['id']} not yet graded. Please grade and re-run")
                    ungraded_cnt += 1
                    del row['rank']
                    ungraded_results.append(row)
            else:
                logger.error(f"Query: {row['query']} not yet graded. Please grade and re-run")
        metrics = {'percent relevant': round((grade_counts['3']/result_cnt)*100, 2),
                   'percent partial': round((grade_counts['2']/result_cnt)*100, 2),
                   'percent notrelevant': round((grade_counts['1']/result_cnt)*100, 2),
                   'average dcg': round(np.mean(dcg_scores), 3),
                   'total queries': len(queries),
                   'total ungraded': ungraded_cnt,
                   'total results': result_cnt}
        if ungraded_cnt > 5:
            data_dir = "./data"
            to_do_filepath = os.path.join(data_dir, "TO-DO_" + os.path.basename(model_results_file))
            logger.warning("\nSearch results need to be graded. \nPlease: grade results in {to_do_filepath}, \nadd to {results_true_file}, \nre-run evaluation")
            with open(to_do_filepath, 'w', encoding='utf-8') as todo:
                columns = ['query','grade','id','snippet','tags','notes']
                writer = csv.DictWriter(todo, fieldnames=columns)
                writer.writeheader()
                for row in ungraded_results:
                    writer.writerow(row)
            logger.info(f"evaluate: wrote results to be manually graded to {to_do_filepath}")
        else:
            logger.info("evaluate: Successfully completed evaluation...")
    return metrics


if __name__ == '__main__':
    data_dir = "./data"
    results_true_file = os.path.join(data_dir, "health_search_results_true.csv")
    results_model_file = os.path.join(data_dir, "health_exchange__tags_snippet_t_t_t_f__results.csv")
    get_metrics(results_true_file=results_true_file, results_pred_file=results_model_file)