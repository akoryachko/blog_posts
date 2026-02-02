VENV = .venv
PROJECT_NAME = pyspark_to_production_blogpost

install:
	curl -LsSf https://astral.sh/uv/install.sh | sh

init: install
	uv tool install ruff
	uv sync
# adding a kernel to use in Jupyter notebook
# needs a restart of VSCode and local environment activation to become visible
	(\
	.  $(VENV)/bin/activate;\
	ipython kernel install --user --name=$(PROJECT_NAME);\
	)

clean:
	rm -rf $(VENV)

# code quality checks
qc:
	ruff check --fix
	ruff format

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

# runs the tip amount model script with parameters in ARGS.
# Syntax: make run ARGS="--n-first-daily-rides-to-keep 5 --test-fraction 0.01"
run:
	(\
	.  $(VENV)/bin/activate;\
	cd ..;\
	python -m pyspark_to_production.src.tip_amount_model ${ARGS};\
	)
