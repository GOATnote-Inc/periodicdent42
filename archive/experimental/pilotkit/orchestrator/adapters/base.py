from __future__ import annotations

import abc
from typing import Iterable

from ..models.schemas import WorkflowEvent


class BaseAdapter(abc.ABC):
    """Interface for workflow data adapters."""

    @abc.abstractmethod
    def stream(self) -> Iterable[WorkflowEvent]:
        raise NotImplementedError
