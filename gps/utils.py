from typing import Iterator, Any


def chunks(li: list, chunk_size: int) -> Iterator[Any]:
    for i in range(0, len(li), chunk_size):
        yield li[i:i + chunk_size]
