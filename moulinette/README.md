## Installation

```bash
uv sync
```

### 1. Generate exercises

```bash
cd src_project_1
uv run python -m moulinette prepare_exercises
```


### 2. Evaluate answers

```bash
uv run python -m moulinette evaluate_student_answers 
```

You can optionally indicate the path to the output :

```bash
uv run python -m moulinette evaluate_student_answers --student_answer_path "../correcs/lbrusa/output/function_calling_name.json"
```

## Modifying the exercises

Edit `moulinette/functions_definition.py` to change the set of generated exercises.