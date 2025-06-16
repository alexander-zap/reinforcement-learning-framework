from typing import Generator, Generic, Sized, TypeVar

T = TypeVar("T")


class SizedGenerator(Generator[T, None, None], Sized, Generic[T]):
    def __init__(self, generator: Generator, size: int, looping: bool):
        self.generator = generator
        self.total_number_of_elements_in_generator = size
        self.number_of_elements_in_current_generator_loop = size
        self.looping = looping

    def __len__(self):
        return self.number_of_elements_in_current_generator_loop

    def __iter__(self):
        return self

    def __next__(self) -> T:
        next_element = next(self.generator)
        self.number_of_elements_in_current_generator_loop -= 1
        if self.number_of_elements_in_current_generator_loop <= 0:
            if self.looping:
                self.number_of_elements_in_current_generator_loop = self.total_number_of_elements_in_generator
            else:
                raise StopIteration
        return next_element

    def send(self, value):
        return self.generator.send(value)

    def throw(self, typ, val=None, tb=None):
        return self.generator.throw(typ, val, tb)

    def close(self):
        return self.generator.close()
