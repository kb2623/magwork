BUILD_DEFINITION_FILE:=env.def
IMAGE_FILE:=env.sif
CONTAINER_NAME:=env

PYTHON_VERSION_MAJOR=3
PYTHON_VERSION_MINOR_FIRST=12
PYTHON_VERSION_MINOR_SECOND=2

SINGULARITY_CACHEDIR=$(pwd)/build

all: build start

build: $(BUILD_DEFINITION_FILE)
	singularity build --build-arg python_version_major=$(PYTHON_VERSION_MAJOR) --build-arg python_version_minor_first=$(PYTHON_VERSION_MINOR_FIRST) --build-arg python_version_minor_second=$(PYTHON_VERSION_MINOR_SECOND) --fakeroot $(IMAGE_FILE) $(BUILD_DEFINITION_FILE)

start: $(IMAGE_FILE)
	singularity instance start $(IMAGE_FILE) $(CONTAINER_NAME)

stop: $(shell singularity instance list $(CONTAINER_NAME))
	singularity instance stop $(CONTAINER_NAME)

clean:
	rm -f ${IMAGE_FILE}
