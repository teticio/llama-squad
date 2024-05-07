#!/usr/bin/env bash
docker-compose run llama-squad bash -c "huggingface-cli login && ./train.sh"
