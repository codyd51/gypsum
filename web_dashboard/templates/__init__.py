import os
from dataclasses import dataclass

import jinja2

JINJA_ENVIRONMENT = jinja2.Environment(loader=jinja2.FileSystemLoader(os.path.dirname(__file__)), autoescape=True)


@dataclass
class TemplateContext:
    """Context common to all templates"""
    generated_at: str

    def __init__(self, **kwargs) -> None:
        self.__dict__.update(**kwargs)
