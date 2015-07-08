#!/bin/bash
sudo docker stop kaggle-avito-ctr
sudo docker rm kaggle-avito-ctr
sudo docker run -d --name kaggle-avito-ctr -e TERM=linux -v /home/$USER/.bashrc:/root/.bashrc:ro -v $PWD/docker/var/lib/postgresql/data:/var/lib/postgresql/data -v $PWD/data:/avito/data -v $PWD/scripts:/avito/scripts -p 5432:5432 postgres
