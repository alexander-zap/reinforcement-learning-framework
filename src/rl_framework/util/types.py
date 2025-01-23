from typing import Generator, Generic, Sized, TypeVar

T = TypeVar("T")


class SizedGenerator(Generator[T, None, None], Sized, Generic[T]):
    def __init__(self, generator: Generator, size: int):
        self.generator = generator
        self.size = size

    def __len__(self):
        return self.size

    def __iter__(self):
        return self

    def __next__(self):
        next_element = next(self.generator)
        self.size -= 1
        return next_element

    def send(self, value):
        return self.generator.send(value)

    def throw(self, typ, val=None, tb=None):
        return self.generator.throw(typ, val, tb)

    def close(self):
        return self.generator.close()
