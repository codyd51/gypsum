from typing import Any, TypeVar, Collection, Iterator

_IterType = TypeVar("_IterType")


def chunks(li: Collection[_IterType], chunk_size: int) -> Iterator[_IterType]:
    for i in range(0, len(li), chunk_size):
        yield li[i:i + chunk_size]


def round_to_previous_multiple_of(val: int, multiple: int) -> int:
    return val - (val % multiple)
