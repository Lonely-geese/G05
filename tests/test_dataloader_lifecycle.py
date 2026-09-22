"""Exercise real worker processes, including early exit with queued tensors."""

from contextlib import closing

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

from utils.dataloader_lifecycle import ManagedDataLoaderIterator
from utils.train_eval import PeriodicEvaluator


def make_loader(workers, persistent, pin_memory=False):
    return DataLoader(
        TensorDataset(torch.arange(128 * 64).reshape(128, 64)),
        batch_size=4,
        num_workers=workers,
        persistent_workers=persistent,
        pin_memory=pin_memory,
        prefetch_factor=2 if workers else None,
    )


@pytest.mark.parametrize("workers,persistent", [(0, False), (2, False), (2, True)])
@pytest.mark.parametrize("fail", [False, True])
def test_early_exit_stops_workers_and_allows_reuse(workers, persistent, fail):
    loader = make_loader(workers, persistent)
    managed = ManagedDataLoaderIterator(loader)
    assert managed._iterator is None
    processes = []
    try:
        with managed:
            assert next(managed)[0].shape == (4, 64)
            processes = list(getattr(managed._iterator, "_workers", []))
            if fail:
                raise ValueError("training failed")
    except ValueError as error:
        assert str(error) == "training failed"
        assert fail
    assert all(not process.is_alive() for process in processes)
    assert managed._iterator is None
    assert loader._iterator is None
    managed.close()
    with managed:
        assert next(managed)[0][0, 0].item() == 0


@pytest.mark.parametrize("persistent", [False, True])
def test_epoch_reset_reuses_only_persistent_workers(persistent):
    loader = make_loader(2, persistent)
    with ManagedDataLoaderIterator(loader) as managed:
        next(managed)
        old = list(managed._iterator._workers)
        managed.reset()
        assert next(managed)[0][0, 0].item() == 0
        new = list(managed._iterator._workers)
        if persistent:
            assert [p.pid for p in old] == [p.pid for p in new]
        else:
            assert all(not p.is_alive() for p in old)
            assert [p.pid for p in old] != [p.pid for p in new]
    assert all(not p.is_alive() for p in new)


def test_evaluator_is_lazy_and_closes_on_exception():
    loader = make_loader(2, True)
    sampler = DistributedSampler(loader.dataset, num_replicas=1, rank=0)
    evaluator = PeriodicEvaluator(loader, sampler, None, None, None)
    assert loader._iterator is None
    with pytest.raises(RuntimeError, match="evaluation failed"):
        with closing(evaluator):
            for _ in range(len(loader) + 1):
                evaluator._next_batch()
            assert sampler.epoch == 1
            processes = list(evaluator._iter._iterator._workers)
            raise RuntimeError("evaluation failed")
    assert all(not p.is_alive() for p in processes)
    assert loader._iterator is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA pin-memory thread required")
@pytest.mark.parametrize("persistent", [False, True])
def test_pin_memory_thread_is_joined_before_exit(persistent):
    loader = make_loader(2, persistent, pin_memory=True)
    # Repeat with outstanding prefetched tensors to exercise shutdown races.
    for _ in range(3):
        with ManagedDataLoaderIterator(loader) as managed:
            assert next(managed)[0].is_pinned()
            pin_thread = managed._iterator._pin_memory_thread
            processes = list(managed._iterator._workers)
        assert not pin_thread.is_alive()
        assert all(not p.is_alive() for p in processes)
