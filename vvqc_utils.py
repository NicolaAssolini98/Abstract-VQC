from interval import interval
import numpy as np

def mc_attack(input_interval, avqc, sample_size):
    # sample a random input from each interval in the input_interval
    for n in range(sample_size):
        print("sample: ", n)
        random_input = [np.random.uniform(input_interval[i][0][0], input_interval[i][0][1]) for i in
                        range(len(input_interval))]

        # compute the output of the circuit for the random input
        # create a closed interval representation for the input
        interval_input = [interval([random_input[i], random_input[i]]) for i in range(len(random_input))]

        # compute the output of the circuit for the random input
        if verify_interval(interval_input=interval_input, avqc=avqc, expected_output=0) == 'unsafe':
            return 'unsafe'

    return 'safe'

# P0 - P1 + b > 0 => 0 sse P0 > P1 - b
# P0 - P1 + b < 0 => 1 sse P0 < P1 - b

def verify_interval(interval_input, avqc, expected_output):
    prob_0, prob_1 = avqc(interval_input)

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


def iterative_refinement(node, heuristic=['random', 'middle']):
    """
    Method to refine the interval node. Using the heuristic defined, it selects a feature to refine and returns two new nodes.
    :param node: list of intervals
    :param heuristic: list of strings, first element is the feature selection heuristic, second element is the split position heuristic
    :return: list of two nodes
    """

    feature_to_split = np.random.randint(0, len(node) - 1)
    split_position = (node[feature_to_split][0][1] + node[feature_to_split][0][0]) / 2

    node_left = node.copy()
    node_right = node.copy()

    node_left[feature_to_split] = interval([node[feature_to_split][0][0], split_position])
    node_right[feature_to_split] = interval([split_position, node[feature_to_split][0][1]])

    return node_left, node_right


def verify(avqc, original_input, epsilon, expected_output, max_depth=50, use_mc_attack=False):
    root = [interval([original_input[i] - epsilon, original_input[i] + epsilon]) for i in range(len(original_input))]
    # TODO: Original input can be a matrix!!
    #  Questo funziona, bisogna implementare il refinement con le matrici (oppure mettere un reshape nel modello astratto di 3_digits_CCQC)
    '''
    root = np.empty(original_input.shape, dtype=object)
    for idx, x in np.ndenumerate(original_input):
        root[idx] = interval([x-epsilon, x + epsilon])
    '''

    frontier = [root]
    next_frontier = []
    depth = 0

    while depth < max_depth:
        depth += 1
        print(f"Depth: {depth}/{max_depth}")
        print(f"\tLen frontier: {len(frontier)}")

        for node in frontier:


            # def verify_interval(interval_input, avqc, expected_output):
            verification_result = verify_interval(interval_input=node, avqc=avqc, expected_output=expected_output)

            if verification_result == 'unsafe':
                return verification_result

            elif verification_result == 'unknown':
                if use_mc_attack and mc_attack(node, avqc, sample_size=1000) == 'unsafe':
                    return 'unsafe'
                node_left, node_right = iterative_refinement(node, heuristic=['random', 'middle'])
                next_frontier.append(node_left)
                next_frontier.append(node_right)

        if frontier == []:
            return 'safe'

        frontier.clear()
        frontier = next_frontier.copy()
        next_frontier.clear()

    return 'unknown'


