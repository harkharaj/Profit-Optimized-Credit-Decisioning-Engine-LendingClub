# On Windows without `make`, run the python commands directly.
PYTHON ?= python

.PHONY: setup data all test app notebook clean

setup:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .
	$(PYTHON) -c "from credit_engine.data import find_raw_file; from credit_engine.config import load_config; print('Raw file found:', find_raw_file(load_config()))"

all:
	$(PYTHON) -m credit_engine.run_all

test:
	$(PYTHON) -m pytest -q

app:
	streamlit run app/streamlit_app.py

notebook:
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace notebooks/credit_decisioning_walkthrough.ipynb

clean:
	rm -rf data/processed/* reports/metrics/* reports/figures/* models/*
