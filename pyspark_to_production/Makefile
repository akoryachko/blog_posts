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

qc:
	ruff check --fix src/
	ruff format src/
