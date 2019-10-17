.PHONY: build build-model inspect-image run-daemon kill-daemon test train-local train-docker clean pipeline gcp-push

clean:
	rm -rf ./cortex

# build:
# 	s2i build -c . c12e/cortex-s2i-daemon-python36-slim:1.0-SNAPSHOT c12e/bank-marketing-daemon

build:
	s2i build -c . c12e/cortex-s2i-model-python36-slim:1.0-SNAPSHOT c12e/bank-marketing-model

inspect-image:
	docker run -it --rm c12e/bank-marketing-model /bin/bash

run-daemon:
	docker run -d --rm --name bank-marketing-model -p 5111:5000 -v $(PWD)/cortex:/model/cortex c12e/bank-marketing-model

kill-daemon:
	docker kill bank-marketing-daemon

run:
	docker run -it --rm -p 5111:5000 c12e/bank-marketing-model

test-endpoint:
	time http -v -j POST :5111/bank-marketing/predict/random-forest < ./test/2-instances.json
	time http -v -j POST :5111/bank-marketing/predict/decision-tree < ./test/2-instances.json

test:
	pytest -s

train-local:
	python model/train.py

train-docker:
	docker run -e TRAIN=1 --rm -v $(PWD)/cortex:/model/cortex -v $(PWD)/data:/model/data c12e/bank-marketing-model

gcp-push:
	docker tag c12e/bank-marketing-model gcr.io/innovation-lab-sandbox/bank-marketing-model
	docker push gcr.io/innovation-lab-sandbox/bank-marketing-model

pipeline:
	dsl-compile --py pipeline/pipeline.py --output bank-marketing-model-pipeline.tar.gz
