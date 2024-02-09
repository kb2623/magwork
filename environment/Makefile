BUILD_DEFINITION_FILE:=env.def
IMAGE_FILE:=env.sif
CONTAINER_NAME:=env

SINGULARITY_CACHEDIR=$(pwd)/build

TEMP_FILE := $(shell mktemp)

all: build start

build: $(BUILD_DEFINITION_FILE)
	singularity build --fakeroot $(IMAGE_FILE) $(BUILD_DEFINITION_FILE)
	rm -f $(TEMP_FILE)

start: $(IMAGE_FILE)
	singularity instance start $(IMAGE_FILE) $(CONTAINER_NAME)

stop: $(shell singularity instance list $(CONTAINER_NAME))
	singularity instance stop $(CONTAINER_NAME)

clean:
	rm -f ${IMAGE_FILE}
