VENV = .venv
PROJECT_NAME = pyspark_to_production_blogpost

install:
	curl -LsSf https://astral.sh/uv/install.sh | sh

init: install
	uv tool install ruff
	uv sync
# adding a kernel to use in Jupyter notebook
# needs a restart of VSCode to become visible
	(\
	.  $(VENV)/bin/activate;\
	ipython kernel install --user --name=$(PROJECT_NAME);\
	)

clean:
	rm -rf $(VENV)

# code quality checks
qc:
	ruff check --fix src/
	ruff format src/

# unit tests run
test:
	(\
	.  $(VENV)/bin/activate;\
	python -m pytest tests -vv -s;\
	)

# runs a single test file/function. Syntax: make test-one ARGS=tests/path_to_file.py::test_function_name
test-one:
	(\
	.  $(VENV)/bin/activate;\
	python -m pytest ${ARGS} -vv -s;\
	)

debug:
	(\
	.  $(VENV)/bin/activate;\
	python -m pytest --pdb -vv -s; \
	)

debug-one:
	(\
	.  $(VENV)/bin/activate;\
	python -m pytest --pdb ${ARGS} -vv -s; \
	)
