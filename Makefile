PYTHON ?= python
PIP ?= pip
CONFIG ?= nexa_train/configs/baseline.yaml

.PHONY: install lint test format train evaluate package clean

install:
	${PIP} install -r requirements.txt

lint:
	${PYTHON} -m ruff src tests
	${PYTHON} -m mypy src

test:
	${PYTHON} -m pytest tests -q

train:
	${PYTHON} scripts/cli.py train --config ${CONFIG}

evaluate:
	${PYTHON} scripts/cli.py evaluate --config ${CONFIG}

package:
	${PYTHON} scripts/cli.py package --config ${CONFIG} --output artifacts/package

clean:
	rm -rf artifacts logs .pytest_cache __pycache__
