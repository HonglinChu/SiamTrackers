# Reformatting

It is recommended to reformat the code before committing it. Here is some useful commands for code reformatting (need yapf / isort / autoflake to be installed).

* _check_ means "only show change, not apply".
* _apply_ means "apply directly"

## Currently in use

Reformatting processes that must be perform before push, otherwise CI will fail.

### yapf

```Bash
# check
yapf -p -r -d --style='{COLUMN_LIMIT:80}' -e "videoanalyst/evaluation/*" -e "debug/*" ./
# apply
yapf -p -r -i --style='{COLUMN_LIMIT:80}' -e "videoanalyst/evaluation/*" -e "debug/*" ./
```

## Currently not in use

Plan to be added to CI in future.

### isort

Order is defined in _video_analyst/.isort.cfg_

```Bash
# check
isort -rc -w 80 -d ./videoanalyst
isort -rc -w 80 -d ./main
# apply
isort -rc -w 80 ./videoanalyst
isort -rc -w 80 ./main
```

### flake

```Bash
# check
autoflake -r --exclude "**/evaluation/**" ./videoanalyst
autoflake -r ./main
# apply
autoflake -r -i --exclude "**/evaluation/**" ./videoanalyst
autoflake -r -i ./main
```

