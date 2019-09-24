# Propensity To Buy Model
![atx](https://img.shields.io/badge/ATX-645-green?style=for-the-badge&logo=graphql)

Demonstrates how to predict propensity to buy using a well known bank marketing dataset. The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit (variable y).

## Prerequisites
1. Python 3.6+
2. [Source-to-Image](https://github.com/openshift/source-to-image)
3. Make

## Quick Start
1. Install Python dependencies:
```
pip install -r requirements.txt
```
2. Train the model(s):
```
make train
```
3. Run unit tests:
```
make test
```
4. Build the Docker image:
```
make build
```

### References
[Bank Marketing Dataset from UCI](http://archive.ics.uci.edu/ml/datasets/Bank+Marketing#)
