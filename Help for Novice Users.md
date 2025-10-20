## How to Install Python
Download Python from [here](https://www.python.org/downloads). Run the installer and make sure to Add Python to PATH (in environment variables). Then open Command Prompt (CMD) and verify installation:
```bash
python --version
```

## How to Create a Virtual Environment
Open Command Prompt (CMD) in your project folder and run the following commands step by step:

1. If you are not in your project folder, navigate to project folder:
```bash
cd path\to\your\project
```
2. Create a virtual environment named `venv`:
```bash
python -m venv venv
```
3. Activate the virtual environment:  

**Windows:**
```bash
venv\Scripts\activate
```
**macOS/Linux:**
```bash
source venv/bin/activate
```
4. Deactivate when done:
```bash
deactivate
```

## Installing Python Packages
While the virtual environment is active, you can install packages:
```bash
pip install package_name
```

## How to Install and Run Your Project
From the project root (where `pyproject.toml` or `requirements.txt` resides):
```bash
pip install -e .
```

### Running Tests
To run all test files in the `tests` directory (using `pytest`):
```bash
pytest
```
To run a specific test file (for example, `test_model.py`):
```bash
pytest tests/test_model.py
```
To run a specific test function inside a test file (for example, `test_train_function` in `test_model.py`):
~~~bash
pytest tests/test_model.py::test_train_function
~~~

✅ **Tip:** Always activate your virtual environment in CMD before installing packages or running scripts. Happy coding! 🚀
