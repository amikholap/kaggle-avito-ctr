#!/bin/bash

CONTAINER_NAME=kaggle-avito-ctr

sudo docker ps | grep -q $CONTAINER_NAME
if [[ $? -eq 0 ]] ; then
    sudo docker exec -it $CONTAINER_NAME gosu postgres psql -d avito
else
    echo "Run $CONTAINER_NAME first."
fi
