.PHONY: all lint test install dev clean distclean

PYTHON ?= python

all: ;

lint:
	q2lint
	flake8

test: all
	py.test -v

install: all
	pip install .

dev: all
	pip install -e .

clean: distclean

distclean: ;
