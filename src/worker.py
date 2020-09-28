# Define function for evaluating each edge connecting to X_n
def edge_eval(x_c, x_n, model, estimator):
    copy = model.copy()
    copy.add_edge(x_c, x_n)
    return estimator.score(copy)