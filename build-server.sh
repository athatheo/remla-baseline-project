#!/bin/bash

docker build -t remlaserver:latest -f Dockerfile-server . \
&& docker tag remlaserver:latest ngavalas/remlaserver:latest \
&& docker push ngavalas/remlaserver:latest
