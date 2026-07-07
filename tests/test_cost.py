import numpy as np

from train.cost_measurement import measure_layer_cost


def test_sparse_and_dense_times_mask_forward_agree_numerically():
    row = measure_layer_cost(in_features=32, out_features=16, sparsity=0.8, n_samples=20, n_repeats=2, seed=0)
    assert row["n_active_weights"] == 32 * 16 - int(round(0.8 * 32 * 16))


def test_flop_accounting_matches_active_weight_count():
    row = measure_layer_cost(in_features=10, out_features=10, sparsity=0.5, n_samples=4, n_repeats=1, seed=0)
    assert row["sparse_flops"] == 2 * 4 * row["n_active_weights"]
    assert row["dense_flops"] == 2 * 4 * 10 * 10
