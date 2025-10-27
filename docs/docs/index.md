# MLOps Project documentation!

## Description

Proyecto de MLOps para la materia TC5044.10

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `aws s3 sync` to recursively sync files in `data/` up to `s3://s3://dvc-mna-mlops-equipo-29-datos-projecto/data/`.
* `make sync_data_down` will use `aws s3 sync` to recursively sync files from `s3://s3://dvc-mna-mlops-equipo-29-datos-projecto/data/` to `data/`.


