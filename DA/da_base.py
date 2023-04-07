from abc import ABCMeta, abstractmethod


class da_base(metaclass=ABCMeta):
    @abstractmethod
    def data_augmentation(self, data, labels):
        pass
