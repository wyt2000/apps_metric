import itertools
import numpy as np
from typing import Dict
from datasets import load_dataset
import testing_util as test_util


DATASET = "codeparrot/apps"


def evaluate_generations(generations, level=["all"]):
    """We take the list of code generations and try to compile them
     and the run their corresponding unit tests which are retrieved from the APPS dataset.

    Args:
        generations: list of code generations, in the same order as APPS dataset samples
        level: list of levels to evaluate, can be "all", "introductory", "interview" or "competition"

    Returns:
        results: dictionary of results, key is the problem index, value is a list of results for each generation
        [-2] = compile error, [-1] = runtime error [False] = failed test case [True] = passed test case
     """

    # generations are code generations in the same order of the dataset
    apps_eval = load_dataset(DATASET, split="test", difficulties=level)
    gpt_codes = generations
    results = {}
    for index in range(len(generations)):
        print(f"task {index}")
        generated_code = gpt_codes[index]
        sample = apps_eval[index]
        res = []
        # loop over the generations
        for o_idx, o in enumerate(generated_code):
            curr_res = [-2]
            try:
                print("Run test")
                curr_res = test_util.run_test(sample, test=o, debug=False)
                print("\nSuccessful compilation!")
                fixed = []
                for e in curr_res:
                    if isinstance(e, np.ndarray):
                       e = e.item(0)
                    if isinstance(e, np.bool_):
                        e = bool(e)
                    fixed.append(e)
                curr_res = fixed
                if not np.all(curr_res):
                    print(f"Results were not True for all test cases") #{curr_res}")
            except Exception as e:
                print(f"Compilation failed, test framework exception = {repr(e)}{e}\n")
                break
            finally:
                assert isinstance(curr_res, list)
                res.append(curr_res)
        results[index] = res

    return results


def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def get_results(results: Dict, count_errors: bool = False, k_list: list = [1, 10, 100]):
    """
    Given the results evaluated against the testcases we output some statistics.
    For single generations:
    >>> example_results = {"0": [[-2]],"1": [[False,False]],"2": [[True,True]],"3": [[False,True,False,True]], "4": [[-1,-1]]}
    >>> get_results(example_results, count_errors=True)
    number of compile errors = 1 avg = 0.2
    number of runtime errors = 1 avg = 0.2
    number of test cases run = 5
    Test Case Average (average accuracy over problems) = 0.3
    Strict Accuracy (all test cases passed / total problems) = 0.2

    For multiple generations:
    >>> example_results = {"0": [[-2], [True, True, True]],"1": [[-1,-1, -1], [True, False, True]]}
    >>> get_results(example_results k_list=[1, 2])
    {'pass@1': 0.25, 'pass@2': 0.5}
    """

    metrics = {"avg_accuracy": None, "strict_accuracy": None, "pass_at_k": None}

    if len(results["0"]) == 1:
        # for single generations we compute average accuracy and stric accuracy: original APPS metrics
        print("Computing accuracy metrics...")
        res = []
        per_prob_res = []
        all_correct = []
        for index in results:
            results[index] = np.array(results[index])
            res.extend(results[index])
            per_prob_res.append(np.mean(results[index]>0))
            all_correct.append(np.all(results[index]>0))
        # we count campilation and runtime errors once per pronlem
        compile_errors = len([e for e in res if -2 in e])
        runtime_errors = len([e for e in res if -1 in e])
        total_testcases = len(res)
        if count_errors:
            print(f"number of compile errors = {compile_errors} avg = {compile_errors / total_testcases}")
            print(f"number of runtime errors = {runtime_errors} avg = {runtime_errors / total_testcases}")
            print(f"number of problems evaluated = {total_testcases}")

        print(f"Test Case Average Accuracy (ver tests) = {np.mean(per_prob_res)}")
        print(f"Strict Accuracy (over problems that pass all tests) = {np.mean(all_correct)}")
        metrics["avg_accuracy"] = np.mean(per_prob_res)
        metrics["strict_accuracy"] = np.mean(all_correct)

    else:
        # for multiple generations we use pass@k metric used in the HumanEval benchmark
        # we use strict accuracy, a generation is valid if it has to pass all the tests
        print("Computing pass@k metric for multiple generations...")
        # total is list with nb generations per task (task=index)
        # correct is number of generations that passed all tests per task
        total = []
        correct = [] 
        for index in results:
            all_correct = []
            for generation in results[index]:
                gen = np.array(generation)
                all_correct.append(np.all(gen>0))
            total.append(len(all_correct))
            correct.append(sum(all_correct))
        total = np.array(total)
        correct = np.array(correct)
        ks = k_list
        pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean() for k in ks if (total >= k).all()}
        print(pass_at_k)
        metrics["pass_at_k"] = pass_at_k
    return metrics

def compute_metrics(generations, k_list=[1, 10, 100], count_errors=True, level=["all"]):
    """Return metrics for the given generations.
    Args:
        generations: dict of generations, keyed by problem index
        k_list: list of k values to compute pass@k when using multiple generations
        count_errors: whether to count compilation and runtime errors when using single generations
        level: which level difficulty in APPS dataset was used for the given generations
    Returns:
        metrics: dict of metrics  
    """
    results = evaluate_generations(generations, level=level)
    metrics = get_results(results, count_errors=count_errors, k_list=k_list)
    return metrics