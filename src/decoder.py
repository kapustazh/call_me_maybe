class ConstrainedDecoder:
    pass


class ImputFormat:
    function_name: str
    description: str
    parameters: dict[str, dict[str, str]]
    return_value: dict[str, dict[str, str]]


class OutputFormat:
    prompt: str
    name: str
    parameters: object
