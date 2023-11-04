import functools
from typing import TypeVar, Collection, Iterator
import hashlib
import time

import numpy as np

from functools import lru_cache, wraps


_IterType = TypeVar("_IterType")


def chunks(li: Collection[_IterType], chunk_size: int, step: int | None = None) -> Iterator[_IterType]:
    chunk_step = chunk_size
    if step:
        if step <= chunk_size:
            raise ValueError(f'Expected the custom step to be at least a chunk size')
        chunk_step = step
    for i in range(0, len(li), chunk_step):
        # Don't return a final truncated chunk
        if len(li) - i < chunk_size:
            #print(f'breaking because were on the last chunk, len={len(li)}, chunk_size={chunk_size}, i={i}')
            break
        yield li[i:i + chunk_size]


def round_to_previous_multiple_of(val: int, multiple: int) -> int:
    return val - (val % multiple)
