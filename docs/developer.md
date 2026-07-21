# Developer Guide

## Setup Development Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Running Tests
Tests are located in the `tests/` directory and use `pytest`.
```bash
pytest tests/
```

## Code Quality
We enforce PEP8 standards using `black`, `flake8`, and `isort`.
```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
```
Type checking is done via `mypy`:
```bash
mypy src/
```

## Adding a New Model
1. Add the PyTorch implementation in a new module inside `src/models/`.
2. Create a Wrapper class inheriting from `BaseModel` (e.g., `src/models/new_model.py`).
3. Add the model to `configs/models.yaml`.
4. Register the model in `src/core/registry.py`.
5. Update `RouterConfig` execution order if it handles a new degradation type.
