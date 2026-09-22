"""Keep DataLoader shutdown on the owning training thread."""


class ManagedDataLoaderIterator:
    """Lazy iterator with explicit, idempotent worker/pin-thread cleanup.

    PyTorch 2.7 has no public DataLoader.close(). Isolate its private shutdown
    hook here; it joins the pin-memory thread before stopping the workers.
    Call close while Python and the workers' resource sharers are still alive.
    """

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iterator = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._iterator is None:
            self.reset()
        return next(self._iterator)

    def reset(self):
        # Persistent workers should survive epoch changes. Nonpersistent
        # iterators may still have prefetched batches when an epoch is cut short.
        if not self.dataloader.persistent_workers:
            self.close()
        self._iterator = iter(self.dataloader)
        return self

    def close(self):
        iterator = self._iterator
        if iterator is None:
            return
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
        # DataLoader caches persistent iterators; don't leave a closed one there.
        if getattr(self.dataloader, "_iterator", None) is iterator:
            self.dataloader._iterator = None
        self._iterator = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
