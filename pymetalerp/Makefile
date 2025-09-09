
MAKEFLAGS += -s
.PHONY: run


run:
	rm -rf dist *.egg-info
	python -m build
	pip install --force-reinstall --no-cache-dir dist/*.whl

