PYTHON = python3

venv:
	$(PYTHON) -m venv pyspark_training

install: venv
	(\
	. pyspark_training/bin/activate; \
	pip install --upgrade pip; \
	pip install -r requirements.txt; \
	)

clean:
	rm -rf .pyspark_training
