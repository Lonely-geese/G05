"""Small real CUDA/DDP run to validate shutdown alongside ongoing training."""
from contextlib import closing
import os

from accelerate import Accelerator
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from utils.dataloader_lifecycle import ManagedDataLoaderIterator
from utils.train_eval import PeriodicEvaluator


def main():
    accelerator = Accelerator()
    dataset = TensorDataset(torch.randn(4096, 128))
    sampler = DistributedSampler(dataset)

    def loader():
        return DataLoader(dataset, batch_size=8, sampler=sampler, num_workers=4,
                          pin_memory=True, persistent_workers=True, prefetch_factor=2)

    train = loader()
    evaluation = loader()
    evaluator = PeriodicEvaluator(evaluation, sampler, None, None, None)
    model = DistributedDataParallel(torch.nn.Linear(128, 16).to(accelerator.device),
                                    device_ids=[accelerator.local_process_index])
    optimizer = torch.optim.AdamW(model.parameters())
    # Match finetune's context ordering and leave prefetched batches outstanding.
    with closing(evaluator), ManagedDataLoaderIterator(train) as batches:
        data_iter = batches.reset()
        for _ in range(10):
            batch = next(data_iter)[0]
            assert batch.is_pinned()
            optimizer.zero_grad()
            model(batch.to(accelerator.device)).square().mean().backward()
            optimizer.step()
        batch = evaluator._next_batch()[0]
        assert batch.is_pinned()
        with torch.no_grad():
            model(batch.to(accelerator.device))
        iterators = [batches._iterator, evaluator._iter._iterator]
        threads = [it._pin_memory_thread for it in iterators]
        workers = [worker for it in iterators for worker in it._workers]
    assert all(not thread.is_alive() for thread in threads)
    assert all(not worker.is_alive() for worker in workers)
    assert train._iterator is None and evaluation._iterator is None
    accelerator.wait_for_everyone()
    print(f"rank={os.environ['RANK']}: 10 updates + eval; 8 workers and 2 pin threads stopped", flush=True)
    accelerator.end_training()


if __name__ == "__main__":
    main()
