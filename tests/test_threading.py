import torch


def test_bench_image_runs_torch_single_threaded_so_step_by_step_filters_do_not_oversubscribe_cores():
    assert torch.get_num_threads() == 1
