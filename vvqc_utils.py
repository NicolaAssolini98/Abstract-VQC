from interval import interval
import numpy as np


def mc_attack(input_interval, avqc, output_class, sample_size, verbose=False):
    """
    Monte Carlo attack: samples inputs uniformly from each dimension's interval and checks for violations.

    :param input_interval: list of interval objects, one per input dimension
    :param avqc: verification object
    :param output_class: expected output
    :param sample_size: number of Monte Carlo samples
    :param verbose: whether to print per-sample debug output
    :return: 'unsafe' if any sample fails verification, 'safe' otherwise
    """
    # Precompute interval bounds for faster sampling

    lower_bounds = np.array([iv[0][0] for iv in input_interval])
    upper_bounds = np.array([iv[0][1] for iv in input_interval])
    dim = len(lower_bounds)

    for n in range(sample_size):
        if verbose:
            print(f"Sample {n + 1}/{sample_size}")

        random_input = np.random.uniform(lower_bounds, upper_bounds)
        interval_input = [interval([v, v]) for v in random_input]

        # Call verifier
        if verify_interval(interval_input=interval_input, avqc=avqc, expected_output=output_class) == 'unsafe':
            return 'unsafe'

    return 'safe'

# P0 - P1 + b > 0 => 0 sse P0 > P1 - b
# P0 - P1 + b < 0 => 1 sse P0 < P1 - b

def verify_interval(interval_input, avqc, expected_output):
    prob_0, prob_1 = avqc(interval_input)
    # print("input: ", interval_input)
    # print("   exp out: ", expected_output)
    # print("   p0:", prob_0, " p1:", prob_1)
    # print("   ",prob_0 & prob_1)
    # print("   ",prob_0 > prob_1)
    if prob_0 & prob_1 != interval():
        return 'unknown'
    elif prob_0 > prob_1:
        if expected_output == 0:
            return 'safe'
        else:
            return 'unsafe'
    else:
        if expected_output == 1:
            return 'safe'
        else:
            return 'unsafe'


def iterative_refinement(node, heuristic={'node':'random', 'pos': 'middle'}):
    """
    Method to refine the interval node. Using the heuristic defined, it selects a feature to refine and returns two new nodes.
    :param node: list of intervals
    :param heuristic: list of strings, first element is the feature selection heuristic, second element is the split position heuristic
    :return: list of two nodes
    """
    # print(node.shape)
    # exit(0)
    if heuristic['node'] == 'random':
        feature_to_split = np.random.randint(0, len(node) - 1)
    else:
        # identify the largest interval feature and split it
        widths = [iv[0][1] - iv[0][0] for iv in node]
        feature_to_split = int(np.argmax(widths))
        
    split_position = (node[feature_to_split][0][1] + node[feature_to_split][0][0]) / 2

    node_left = node.copy()
    node_right = node.copy()

    node_left[feature_to_split] = interval([node[feature_to_split][0][0], split_position])
    node_right[feature_to_split] = interval([split_position, node[feature_to_split][0][1]])

    return node_left, node_right


def verify(avqc, original_input, epsilon, expected_output, max_depth=50, use_mc_attack=False, verbose=False):

    original_input = original_input.flatten() if isinstance(original_input, np.ndarray) else original_input
    root = [interval([original_input[i] - epsilon, original_input[i] + epsilon]) for i in range(len(original_input))]


    frontier = [root]
    next_frontier = []
    depth = 0

    if verbose:
        print(f"Epsilon: {epsilon}")
    
    while depth < max_depth:
        depth += 1
        tot_safe_nodes = 0

        for node in frontier:
            verification_result = verify_interval(interval_input=node, avqc=avqc, expected_output=expected_output)
           
            if verification_result == 'unsafe':
                return 'unsafe'

            elif verification_result == 'unknown':
                if use_mc_attack and mc_attack(node, avqc, expected_output, sample_size=100) == 'unsafe':
                    return 'unsafe'
            
                node_left, node_right = iterative_refinement(node, heuristic={'node':'larger', 'pos': 'middle'})
                next_frontier.append(node_left)
                next_frontier.append(node_right)
            
            else:
                tot_safe_nodes+=1

        if verbose: 
            print(f"Depth: {depth-1}/{max_depth-1}")
            print(f"\tLen frontier: {len(frontier)}")
            print(f"\t# safe node discovered: {tot_safe_nodes}")
            print(f"\t# unknown nodes: {len(frontier)- tot_safe_nodes}")
            print(f"\t# node next iteration: {len(next_frontier)}\n\n")
        

        if next_frontier == []:
            return 'safe'

        frontier.clear()
        frontier = next_frontier.copy()
        next_frontier.clear()

    return 'unknown'

def compute_maximum_epsilon(avqc, original_input, expected_output, min_epsilon=0.0001, max_epsilon=1, tolerance=1e-4, verbose=True):
    """
    Find the maximum epsilon such that the model's output does not change under perturbation of that size.

    :param verbose: print debug information
    :param avqc: verifier object with a method `verify(input, epsilon)` -> bool
    :param original_input: input to test robustness against
    :param expected_output: expected (correct) output of the model
    :param min_epsilon: minimum epsilon to start search from
    :param max_epsilon: maximum epsilon limit
    :param tolerance: precision of the final epsilon value
    :return: maximum tolerated epsilon
    """
    # exponential search to find an upper bound where robustness fails
    epsilon = min_epsilon
    while epsilon <= max_epsilon and verify(avqc, original_input, epsilon, expected_output, max_depth=8, use_mc_attack=False, verbose=verbose)=='safe':
        epsilon *= 2
    if verbose:
        print(f"Upper bound found: {epsilon/2}")
    
    high = min(epsilon, max_epsilon)
    low = epsilon/2

    # binary search between low and high
    best_eps = low
    while high - low > tolerance:
        mid = (low + high) / 2
        if verify(avqc, original_input, mid, expected_output, max_depth=8, use_mc_attack=False,verbose=verbose)=='safe':
            best_eps = mid
            low = mid
        else:
            high = mid

    return best_eps



