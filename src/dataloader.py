from abc import ABC, abstractmethod


class DataLoader(ABC):
    batch_size: int

    @abstractmethod
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int):
        raise NotImplementedError
