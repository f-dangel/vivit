.DEFAULT: help

help:
	@echo "install"
	@echo "        Install vivit and dependencies"
	@echo "uninstall"
	@echo "        Unstall vivit"
	@echo "install-dev"
	@echo "        Install only the development tools"
	@echo "install-test"
	@echo "        Install only the testing tools (included in install-dev)"
	@echo "test"
	@echo "        Run pytest on test and report coverage"
	@echo "test-light"
	@echo "        Run pytest on the light part of test and report coverage"
	@echo "examples"
	@echo "        Run the examples"
	@echo "install-lint"
	@echo "        Install only the linter tools (included in install-dev)"
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
	@pip install -r requirements.txt
	@pip install .

.PHONY: uninstall

uninstall:
	@pip uninstall vivit


.PHONY: install-dev

install-dev:
	@make install-test
	@make install-lint

.PHONY: install-test

install-test:
	@pip install -r requirements/test.txt

.PHONY: test test-light

test:
	@pytest -vx --run-optional-tests=expensive --cov=vivit test

test-light:
	@pytest -vx --cov=vivit test

.PHONY: examples

examples:
	@cd exp/examples && python example_uncentered_gradient_covariance.py
	@cd exp/examples && python example_centered_gradient_covariance.py
	@cd exp/examples && python example_ggn_exact.py
	@cd exp/examples && python example_optimizer_mnist.py
	@cd exp/examples && python example_toy_problem.py
	@cd exp/examples && python example_efficient_quadratic_toy_problem.py
	@cd exp/examples && python example_directional_derivatives.py
	@cd exp/examples && python example_newton_step.py

.PHONY: install-lint

install-lint:
	@pip install -r requirements/lint.txt

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
