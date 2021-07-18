import time
import itertools, functools
from concurrent.futures import ProcessPoolExecutor
import psutil
import typing

from pgmpy.models import BayesianModel, DynamicBayesianNetwork, DynamicNode
from pgmpy.estimators import BDeuScore
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import State

import pandas as pd
from matplotlib import pyplot as plt
from tqdm import notebook
from IPython.utils import io  # useful to suppress rejection_sampling's output


def plot_signals(df: typing.Union[pd.DataFrame, typing.Sequence[pd.DataFrame]], title: str, 
                 figsize: typing.Optional[typing.Tuple[int, int]] = None):
    _, axes = plt.subplots(nrows=df.shape[1] // 2, ncols=2, figsize=figsize)
    plt.suptitle(title, size='xx-large')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    for i, col in enumerate(df):
        id1 = i // 2
        id2 = i % 2

        df[col].plot(ax=axes[id1, id2], grid=True, title=col)


def plot_train_test_signals(df: pd.DataFrame, title: str, df_train: pd.DataFrame, df_test: pd.DataFrame, 
                            figsize: typing.Optional[typing.Tuple[int, int]] = None):
    _, axes = plt.subplots(nrows=df.shape[1] // 2, ncols=2, figsize=figsize)
    plt.suptitle(title, size='xx-large')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    for i, col in enumerate(df):
        id1 = i // 2
        id2 = i % 2

        x_data = range(len(df))
        axes[id1, id2].plot(x_data[:len(df_train)], df_train[col], 'C1', label='Training set')
        axes[id1, id2].plot(x_data[len(df_train):], df_test[col], 'C2', label='Test set')
        axes[id1, id2].set_title(col)
        axes[id1, id2].legend(loc='upper left')
        axes[id1, id2].grid()


def plot_synth_signal(df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame,
                            title: str, figsize: typing.Optional[typing.Tuple[int, int]] = None):
    _, axes = plt.subplots(nrows=df1.shape[1], ncols=3, figsize=figsize)
    plt.suptitle(title, size='xx-large')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    for i, col in enumerate(df1):
        axes[i, 0].plot(df1[col], 'C0')
        axes[i, 1].plot(df2[col], 'C1')
        axes[i, 2].plot(df3[col], 'C2')
        axes[i, 0].set_title(f'{col} - Original')
        axes[i, 1].set_title(f'{col} - Synthetic')
        axes[i, 2].set_title(f'{col} - Baseline (random)')
        axes[i, 0].grid()
        axes[i, 1].grid()
        axes[i, 2].grid()


def discretize_series(series: pd.Series) -> pd.Series:
    # Find minimum and maximum values of the series, and define boundaries
    s_min, s_max = series.min(), series.max()
    boundaries = [s_min / 2., 0., s_max / 2.]
    
    # Assign tier based on the boundaries
    def get_tier(val):
        if val < boundaries[0]:
            return 'Very Low'
        elif val >= boundaries[0] and val < boundaries[1]:
            return 'Low'
        elif val >= boundaries[1] and val < boundaries[2]:
            return 'High'
        else:
            return 'Very High'
    
    return pd.Series([get_tier(val) for val in series])


def unroll_series(series: pd.Series) -> typing.Tuple[pd.Series, pd.Series]:
    # Copy the original series
    next_series = series.copy()
    series = series[:-1].reset_index(drop=True)
    next_series = next_series[1:].reset_index(drop=True)
    return series, next_series


def edge_eval(x_c: str, x_n: str, model: BayesianModel, estimator: BDeuScore) -> float:
    # Define function for evaluating each edge connecting to X_n
    copy = model.copy()
    copy.add_edge(x_c, x_n)
    return estimator.score(copy)


def forward_stepwise_selection(current_nodes: typing.Sequence[str], next_nodes: typing.Sequence[str], 
                               indegree: int, estimator: BDeuScore, verbose: bool = False) \
                                   -> typing.Tuple[BayesianModel, float]:
    assert indegree <= len(current_nodes), 'Indegree should not be greater than the number of nodes in a'

    start_time = time.time()
    
    # Initialize model with continuity constraints
    continuity_constraints = list(zip(current_nodes, next_nodes))
    model = BayesianModel(continuity_constraints)

    # Create support dictionary to find node ICX_t from ICX_t+1
    node_map = dict([(n, c) for c, n, in continuity_constraints])
    
    # Get number of available cores
    n_procs = psutil.cpu_count(logical=False)
    if verbose:
        print(f'Using {n_procs} cores')
        print('-' * 50)
    
    # If indegree is equal to the total number of nodes in the current slice,
    # speed up the search by simply constructing a dense network
    if indegree == len(current_nodes):
        if verbose:
            print('Building dense network...')
        edges = list(itertools.product(current_nodes, next_nodes))
        model.add_edges_from(edges)
    else:
        # Scan next nodes
        for x_n in next_nodes:
            if verbose:
                print(f'Finding parents of node {x_n}...')
            
            # Keep track of already added nodes to avoid repetitions
            # Take continuity constraints into account in the added nodes
            added_nodes = set()
            added_nodes.add(node_map[x_n])
            # Loop is repeated (indegree - 1) times since one parent node 
            # has already been added due to continuity constraints
            for _ in range(indegree - 1):
                # Consider as testable only non-looping nodes
                testable_nodes = sorted(set(current_nodes).difference(added_nodes))  # sort for reproducibility

                # Use multiprocessing to speed-up search
                with ProcessPoolExecutor(max_workers=n_procs) as executor:
                    # Fix estimator, model and x_n parameters and feed the partial function to the executor
                    edge_eval_par = functools.partial(edge_eval, estimator=estimator, model=model, x_n=x_n)
                    scores = dict(zip(testable_nodes, executor.map(edge_eval_par,
                                                                   testable_nodes)))
                node_to_add = max(scores, key=scores.get)
                best_score = scores[node_to_add]
                model.add_edge(node_to_add, x_n)
                added_nodes.add(node_to_add)
                if verbose:
                    print(f'\t Added {node_to_add} with {best_score:.2f}')
    final_score = estimator.score(model)
    if verbose:
        print('-' * 50)
        print(f'Running time: {time.time() - start_time:.2f} s')
    return model, final_score


def grid_search(indegree_range: typing.Sequence[int], current_nodes: typing.Sequence[str],
                next_nodes: typing.Sequence[str], estimator: BDeuScore) -> int:
    scores = dict()
    for indegree in indegree_range:
        start_time = time.time()
        _, score = forward_stepwise_selection(current_nodes, next_nodes, indegree, estimator)
        scores[indegree] = score
        print(f'Testing indegree={indegree} with score {score:.2f} [{time.time() - start_time:.2f} s]...')
    
    return max(scores, key=scores.get)


def print_edges(model: typing.Union[BayesianModel, DynamicBayesianNetwork]):
    assert isinstance(model, BayesianModel) or isinstance(model, DynamicBayesianNetwork), \
        ('Expected BayesianModel or DynamicBayesianNetwork.')

    groups = itertools.groupby(model.edges(), key=lambda t: t[0])
    sorted_groups = sorted([(parent, sorted([child for child in children])) for parent, children in groups], 
                           key=lambda t: t[0])
    for parent, children in sorted_groups:
        print(parent)
        for child in children:
            print(f'|-- {child[1]}')
    print('-' * 25)
    print(f'Total number of edges: {len(model.edges())}')


def to_dynamic_cpd(static_model: BayesianModel, stat_to_dyn_map: typing.Dict[str, str], 
                   next_to_curr_map: typing.Dict[str, str]) -> TabularCPD:
    # Lambda to obtain dynamic nodes' name
    get_dynamic_node = lambda node: DynamicNode(stat_to_dyn_map[node], 0) if node.endswith('_T') \
        else DynamicNode(stat_to_dyn_map[next_to_curr_map[node]], 1)
    
    # Extract information about CPDs of the static model
    cpds_info = [{'variable': get_dynamic_node(cpd.variable),
                  'variable_card': 4,
                  'values': cpd.get_values(),
                  'evidence': [DynamicNode(stat_to_dyn_map[e], 0) for e in cpd.get_evidence()][::-1] \
                      if len(cpd.get_evidence()) > 0 else None,
                  'evidence_card': [4] * len(cpd.get_evidence()) \
                      if len(cpd.get_evidence()) > 0 else None,
                  'state_names': {get_dynamic_node(k): v for k, v in cpd.state_names.items()}} \
                      for cpd in static_model.get_cpds()]
    
    return [TabularCPD(**cpd_info) for cpd_info in cpds_info]


def get_order(elimination_order: str, infer: VariableElimination, 
              variables: typing.Sequence[str], evidence: typing.Sequence[str]):
    # Wrapper for infer._get_elimination_order:
    # functools.partial fixes function arguments right-to-left, so the elimination order should be the left-most
    order = infer._get_elimination_order(variables=variables,
                                         evidence=evidence,
                                         elimination_order=elimination_order,
                                         show_progress=False)
    # Additionally, execute query to assess performance
    start_time = time.time()
    infer.query(variables=variables, evidence=evidence, elimination_order=order, show_progress=False)
    return order, round(time.time() - start_time, 5)


def compare_heuristics(infer: VariableElimination, variables: typing.Sequence[str], 
                       evidence: typing.Optional[typing.Sequence[str]] = None):
    heuristics = ['MinFill', 'MinNeighbors', 'MinWeight', 'WeightedMinFill']
    n_procs = psutil.cpu_count(logical=False)
    print(f'\nUsing {n_procs} cores')
    print('-' * 50)
    with ProcessPoolExecutor(max_workers=n_procs) as executor:
        # Fix evidence, variables and inference algorithm and feed the partial function to the executor
        get_order_par = functools.partial(
            get_order,
            evidence=evidence,
            variables=variables,
            infer=infer
        )
        orders = dict(zip(heuristics, executor.map(get_order_par,
                                                   heuristics)))

    if (orders['MinFill'][0] == orders['MinNeighbors'][0] 
            and orders['MinFill'][0] == orders['MinWeight'][0] 
            and orders['MinFill'][0] == orders['WeightedMinFill'][0]):
        if evidence is None:
            print(f'All heuristics have found the same elimination order for variables {variables} '
                  f'with no evidence:')
        else:
            print(f'All heuristics have found the same elimination order for variables {variables} '
                  f'with evidence {evidence}:')
    else:
        if evidence is None:
            print(f'The heuristics have found different elimination orders for variables {variables} '
                  f'with no evidence:')
        else:
            print(f'The heuristics have found different elimination orders for variables {variables} '
                  f'with evidence {evidence}:')
    
    for h in heuristics:
        print(f'{h}: {orders[h][0]} [{orders[h][1]} s]')


def exec_query(infer: VariableElimination, variables: typing.Sequence[str], 
               evidence: typing.Optional[typing.Sequence[str]] = None, heuristics:str = 'MinFill'):
    if evidence is None:
        print(f'\nProbability query for {variables} given no evidence computed with {heuristics} heuristics:')
    else:
        print(f'\nProbability query for {variables} given {evidence} computed with {heuristics} heuristics:')
    start_time = time.time()
    print(infer.query(variables=variables, evidence=evidence, elimination_order=heuristics, show_progress=False))
    print(f'Running time: {time.time() - start_time:.5f} s')


def generate_time_series(sampler: BayesianModelSampling, length: int, labels: typing.List[str], seed: int = 42):
    # Initialize progress bar
    pbar = notebook.tqdm(total=length)

    # Generate first sample given no evidence
    with io.capture_output() as captured:
        # When no evidence is provided, the function under-the-hood performs forward sampling
        sample = sampler.rejection_sample(seed=seed)
    sample = sample.reindex(sorted(sample.columns), axis=1)

    # Split sample in 'current' and 'next' slices:
    # - the 'current' slice will be the first row of the generated time series
    # - the 'next' slice is added as the second row, and will be used as
    # evidence for subsequent predictions
    df_synth = sample.filter(regex='_T$')
    next_slice = sample.filter(regex='_T\+1').iloc[0].values.tolist()
    df_synth = df_synth.append(pd.Series(next_slice, index=df_synth.columns), ignore_index=True)
    evidence = [State(n, v) for n, v in zip(df_synth.columns.values, next_slice)]

    # Update progress bar
    pbar.update(2)

    for _ in range(2, length):
        # Generate new data
        with io.capture_output() as captured:
            sample = sampler.rejection_sample(evidence=evidence)
        sample = sample.reindex(sorted(sample.columns), axis=1)

        # Append 'next' slice to the generated time series, and use it as new evidence
        next_slice = sample.filter(regex='_T\+1').iloc[0].values.tolist()
        df_synth = df_synth.append(pd.Series(next_slice, index=df_synth.columns), ignore_index=True)
        evidence = [State(n, v) for n, v in zip(df_synth.columns.values, next_slice)]

        # Update progress bar
        pbar.update(1)
    # Close progress bar
    pbar.close()
    # Update column names
    df_synth.columns = labels
    return df_synth


def active_trail(dynamic_model: DynamicBayesianNetwork, node: typing.Tuple[str, int],
                 evidence: typing.Sequence[typing.Tuple[str, int]] = []):
    node = DynamicNode(*node)
    evidence = [DynamicNode(*e) for e in evidence]
    reachable = dynamic_model.active_trail_nodes(node, observed=evidence).get(node)
    reachable.remove(node)
    reachable = sorted(reachable)
    if reachable:
        if evidence:
            print(f'Active trail between {node} and {reachable} given {evidence}')
        else:
            print(f'Active trail between {node} and {reachable} given no evidence')
    else:
        if evidence:
            print(f'No active trails from {node} given {evidence}')
        else:
            print(f'No active trails from {node} given no evidence')
