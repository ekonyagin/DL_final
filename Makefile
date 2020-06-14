PWD=$$(pwd)

.PHONY: train-local
train-local:
	PYTHONPATH=$(PWD) ROOT_DIR=$(PWD) python src/train.py

.PHONY: sample-local
sample-local:
	PYTHONPATH=$(PWD) ROOT_DIR=$(PWD) python src/sample.py

.PHONY: setup-local
setup-local:
	pip install -r requirements.txt

NEURO=neuro
PROJECT_PATH_STORAGE=storage:DL_final
SETUP_JOB=setup-job
PROJECT_PATH_ENV=/project
TRAIN_JOB=dl-final-train
TRAIN_STREAM_LOGS=yes
RUN?=$$(ROOT_DIR=$$(pwd) python -c 'import hashlib; from config import cfg; print("h" + hashlib.md5(bytes(cfg.EXPERIMENT_TAG.encode("utf-8"))).hexdigest())')
DESCRIPTION=$$(ROOT_DIR=$$(pwd) python -c 'from config import cfg; print(cfg.EXPERIMENT_TAG)')
TRAIN_CMD=make train

test:
	echo $(RUN)

.PHONY: upload-all
upload-all: ### Setup remote environment
	$(NEURO) cp --recursive ./* $(PROJECT_PATH_STORAGE)

.PHONY: upload-config
upload-config:  ### Upload config directory to the platform storage
	$(NEURO) cp \
		--recursive \
		--update \
		--no-target-directory \
		config $(PROJECT_PATH_STORAGE)/config

.PHONY: upload-code
upload-code:  ### Upload config directory to the platform storage
	$(NEURO) cp \
		--recursive \
		--update \
		--no-target-directory \
		src $(PROJECT_PATH_STORAGE)/src

.PHONY: download-results
download-results:  ### Download results directory from the platform storage
		$(NEURO) cp \
		--recursive \
		--update \
        --exclude="*.pt*" \
		--no-target-directory \
		$(PROJECT_PATH_STORAGE)/results results

.PHONY: setup
setup-cluster: ### Setup remote environment
	$(NEURO) mkdir --parents $(PROJECT_PATH_STORAGE)
	$(NEURO) cp --recursive ./* $(PROJECT_PATH_STORAGE)
	$(NEURO) run \
		--name $(SETUP_JOB) \
		--tag "target:setup" \
		--preset cpu-small \
		--detach \
		--life-span=1h \
		--volume $(PROJECT_PATH_STORAGE):$(PROJECT_PATH_ENV):ro \
        neuromation/base:v1.6 \
		'sleep infinity'
	$(NEURO) exec --no-key-check -T $(SETUP_JOB) "bash -c 'pip install --progress-bar=off -U --no-cache-dir -r $(PROJECT_PATH_ENV)/requirements.txt'"
	$(NEURO) --network-timeout 300 job save $(SETUP_JOB) image:dl-final
	$(NEURO) kill $(SETUP_JOB) || :

.PHONY: kill-setup
kill-setup:
	$(NEURO) kill $(SETUP_JOB)

.PHONY: train
train: upload-config upload-code ### Run a training job (set up env var 'RUN' to specify the training job),
#     $(NEURO) cp --recursive file:. $(PROJECT_PATH_STORAGE)
	$(NEURO) run \
		--name $(RUN) \
		--tag "target:train" \
		--preset gpu-small \
		--detach \
		--http 6006 \
		--no-http-auth \
		--volume $(PROJECT_PATH_STORAGE):$(PROJECT_PATH_ENV):rw \
		--volume storage:hologan/data:$(PROJECT_PATH_ENV)/data1:rw \
		--env PYTHONPATH=$(PROJECT_PATH_ENV) \
		--env EXPOSE_SSH=yes \
		--life-span=3d \
		--env JOB_TIMEOUT=0 \
		image:dl-final \
		bash -c 'tensorboard --logdir $(PROJECT_PATH_ENV)/results/ --bind_all & cd $(PROJECT_PATH_ENV) && $(TRAIN_CMD)'
# 		'sleep infinity'
ifeq ($(TRAIN_STREAM_LOGS), yes)
	@echo "Streaming logs of the job $(RUN)"
	$(NEURO) exec --no-key-check -T $(RUN) "tail -f /output" || echo -e "Stopped streaming logs.\nUse 'neuro logs <job>' to see full logs."
endif

.PHONY: kill-train-curr
kill-train-curr:  ### Terminate all training jobs you have submitted
	$(NEURO) kill $(RUN)

.PHONY: kill-train-all
kill-train-all:  ### Terminate all training jobs you have submitted
	jobs=`neuro -q ps --tag "target:train" $(_PROJECT_TAGS) | tr -d "\r"` && \
	[ ! "$$jobs" ] || $(NEURO) kill $$jobs
