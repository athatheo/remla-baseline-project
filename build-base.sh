#!/bin/bash

docker build -t remlabase:latest -f Dockerfile-base . \
&& docker tag remlabase:latest ngavalas/remlabase:latest \
&& docker push ngavalas/remlabase:latest
