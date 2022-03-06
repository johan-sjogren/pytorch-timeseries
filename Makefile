#################################################################################
# COMMANDS                                                                      #
#################################################################################
.PHONY: env
env:
    conda env create -f conda.yaml

.PHONY: init
init:
    conda activate pytorch-ts

## precommit hooks with flak8 and black
.PHONY: precommit
precommit:
	pre-commit run --all

## Format code using flake8
.PHONY: lint
lint:
	flake8 src

## Format your code using black
.PHONY: black
black:
	black . -v

# Run tests using pytest github actions (including integration test)
# .PHONY: test
# test: clean-pyc clean-test
# 	python -m pytest tests

# ## Delete all compiled Python files
# .PHONY: clean-pyc
# clean-pyc:  ## Remove Python file artifacts.
# 	find . -name '*.pyc' -exec rm -f {} +
# 	find . -name '*.pyo' -exec rm -f {} +
# 	find . -name '*~' -exec rm -f {} +
# 	find . -name '__pycache__' -exec rm -fr {} +

# .PHONY: clean-test
# clean-test:  ## Remove test and coverage artifacts.
# 	rm -f .coverage coverage.xml
# 	rm -fr htmlcov/ .junit/
# 	rm -fr .pytest_cache