from enum import Enum
from typing import Any, Dict, List


def group_dict_key_by_value(d_input: Dict[Any, Any]) -> Dict[Any, List[Any]]:
    dict_value_key: Dict[Any, List[Any]] = {}
    for i, v in d_input.items():
        dict_value_key[v] = (
            [i] if v not in dict_value_key.keys() else dict_value_key[v] + [i]
        )

    return dict_value_key


def enum_dict_factory(data: Any) -> Dict[Any, Any]:
    def convert_value(obj: Any) -> Any:
        if isinstance(obj, Enum):
            return obj.value
        return obj

    return dict((k, convert_value(v)) for k, v in data)
