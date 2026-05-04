import inspect
import importlib.util
import json
import typing
from typing import Dict, Any, List, get_type_hints

from pydantic import BaseModel

class FunctionInfo(BaseModel):
    fn_name: str
    args_names: list[str]
    args_types: Dict[str, str]
    return_type: str


def extract_functions_infos(fn: Any) -> FunctionInfo:
    # Get function name
    fn_name = fn.__name__

    # Get argument names from signature
    args_names = list(fn.__code__.co_varnames[:fn.__code__.co_argcount])
    
    # Get type hints including return type
    type_hints = get_type_hints(fn)
    
    # Split into args types and return type
    return_type = type_hints.pop('return').__name__
    args_types = {k: v.__name__ for k,v in type_hints.items()}

    return FunctionInfo(
        fn_name=fn_name,
        args_names=args_names, 
        args_types=args_types,
        return_type=return_type
    )

def get_all_functions_from_module(module_path: str) -> List[FunctionInfo]:
    """
    Extract information about all functions defined in a module.
    
    Args:
        module_path: Path to the Python module file
        
    Returns:
        List of FunctionInfo objects for each function defined in the module
    """
    # Load module from path
    spec = importlib.util.spec_from_file_location("dynamic_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get all module members
    members = inspect.getmembers(module)
    
    # Filter for functions defined in this module
    functions = [
        member[1] for member in members 
        if inspect.isfunction(member[1]) 
        and member[1].__module__ == "dynamic_module"
    ]
    return functions
    

def generate_function_calling_definition(
    import_module_path: str,
    output_json_path: str,
) -> None:
    all_functions = get_all_functions_from_module(import_module_path)

    function_infos = [extract_functions_infos(fn) for fn in all_functions]

    function_infos_list = [fn.model_dump() for fn in function_infos]

    with open(output_json_path, "w") as f:
        json.dump(function_infos_list, f, indent=2)
