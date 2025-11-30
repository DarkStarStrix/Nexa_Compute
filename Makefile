PYTHON ?= python
PIP ?= pip
CONFIG ?= nexa_train/configs/baseline.yaml

.PHONY: install lint test format train evaluate package clean

install:
	${PIP} install -r requirements.txt

lint:
	${PYTHON} -m ruff check src tests
	${PYTHON} -m mypy src

test:
	${PYTHON} -m pytest tests -q

train:
	${PYTHON} src/nexa_compute/cli/main.py launch --config ${CONFIG}

evaluate:
	${PYTHON} src/nexa_compute/cli/main.py evaluate --config ${CONFIG}

package:
	${PYTHON} src/nexa_compute/cli/main.py build-containers

clean:
	rm -rf artifacts logs .pytest_cache __pycache__
