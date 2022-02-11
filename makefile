.DEFAULT: help

help:
	@echo "install"
	@echo "        Install vivit and dependencies"
	@echo "uninstall"
	@echo "        Unstall vivit"
	@echo "docs"
	@echo "        Build the documentation"
	@echo "install-dev"
	@echo "        Install vivit and development tools"
	@echo "install-docs"
	@echo "        Install vivit and documentation tools"
	@echo "install-test"
	@echo "        Install vivit and testing tools"
	@echo "test"
	@echo "        Run pytest on test and report coverage"
	@echo "install-lint"
	@echo "        Install vivit and the linter tools"
	@echo "isort"
	@echo "        Run isort (sort imports) on the project"
	@echo "isort-check"
	@echo "        Check if isort (sort imports) would change files"
	@echo "black"
	@echo "        Run black on the project"
	@echo "black-check"
	@echo "        Check if black would change files"
	@echo "flake8"
	@echo "        Run flake8 on the project"
	@echo "conda-env"
	@echo "        Create conda environment 'vivit' with dev setup"
	@echo "darglint-check"
	@echo "        Run darglint (docstring check) on the project"
	@echo "pydocstyle-check"
	@echo "        Run pydocstyle (docstring check) on the project"

.PHONY: install

install:
	@pip install -e .

.PHONY: uninstall

uninstall:
	@pip uninstall vivit

.PHONY: docs

docs:
	@cd docs/rtd && make html
	@echo "\nOpen docs/rtd/index.html to see the result."

.PHONY: install-dev

install-dev: install-lint install-test install-docs

.PHONY: install-docs

install-docs:
	@pip install -e .[docs]

.PHONY: install-test

install-test:
	@pip install -e .[test]

.PHONY: test test-light

test:
	@pytest -vx --cov=vivit test

.PHONY: install-lint

install-lint:
	@pip install -e .[lint]

.PHONY: isort isort-check

isort:
	@isort .

isort-check:
	@isort . --check --diff

.PHONY: black black-check

black:
	@black . --config=black.toml

black-check:
	@black . --config=black.toml --check

.PHONY: flake8

flake8:
	@flake8 .

.PHONY: darglint-check

darglint-check:
	@darglint --verbosity 2 vivit

.PHONY: pydocstyle-check

pydocstyle-check:
	@pydocstyle --count .

.PHONY: conda-env

conda-env:
	@conda env create --file .conda_env.yml
