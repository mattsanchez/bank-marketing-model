.PHONY: build inspect-image run-daemon kill-daemon test train

build:
	s2i build -c . c12e/cortex-s2i-daemon-python36-slim:1.0-SNAPSHOT c12e/bank-marketing-model

inspect-image:
	docker run -it --rm c12e/bank-marketing-model /bin/bash

run-daemon:
	docker run -d --rm --name bank-marketing-model -p 5111:5000 c12e/bank-marketing-model

kill-daemon:
	docker kill bank-marketing-model

run:
	docker run -it --rm -p 5111:5000 c12e/bank-marketing-model

test-endpoint:
	time http -v -j POST :5111/bank-marketing/predict < ./test/1-instance.json

test:
	pytest -s

train:
	python model/train.py
