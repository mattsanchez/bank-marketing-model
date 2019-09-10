.PHONY: build

build:
	s2i build . c12e/cortex-s2i-daemon-python36-slim:1.0-SNAPSHOT c12e/bank-marketing-model
