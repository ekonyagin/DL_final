PWD=$$(pwd)

.PHONY: train
train:
	PYTHONPATH=$(PWD) ROOT_DIR=$(PWD) python src/train.py

.PHONY: sample
sample:
	PYTHONPATH=$(PWD) ROOT_DIR=$(PWD) python src/sample.py

.PHONY: setup
setup:
	pip install -r requirements.txt
