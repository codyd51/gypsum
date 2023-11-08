from dataclasses import dataclass


@dataclass
class Event:
    pass


class UnknownEventError(Exception):
    pass
