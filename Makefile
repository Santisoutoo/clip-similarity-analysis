IMAGE := sii-clip
PROJECT_DIR := $(shell pwd)

# Build Docker image
build:
	docker build -t $(IMAGE) .

shell:
	docker run -it \
		--shm-size=24g \
		-v $(PROJECT_DIR):/opt/project \
		-v $(PROJECT_DIR)/outputs:/opt/project/outputs \
		--rm \
		$(IMAGE) /bin/bash

run-preclip-cpu:
	docker run \
		--shm-size=8g \
		-v $(PROJECT_DIR):/opt/project \
		-v $(PROJECT_DIR)/outputs:/opt/project/outputs \
		--rm \
		$(IMAGE) python /opt/project/src/pre-clip/main.py