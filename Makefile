.PHONY: install install-dev test build clean

install:
	pip install poetry==1.8.4
	poetry install

###################
# Developer tools #
###################

install-dev:
	pip install poetry==1.8.4
	poetry install --with dev

test:
	cd tests && poetry run pytest -s

build:
	poetry build

clean:
	rm -rf build dist *.egg-info
