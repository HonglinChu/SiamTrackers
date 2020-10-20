# SAT

## Training

The config files are in examples/sat/train

```Bash
python3 ./main/dist_train_sat.py --config 'path/to/config.yaml'
python3 ./main/test.py --config 'path/to/config.yaml'
```

Resuming from epoch number

```Bash
python3 ./main/dist_train_sat.py --config 'path/to/config.yaml' --resume 10
```

Resuming from certain snapshot file

```Bash
python3 ./main/dist_train_sat.py --config 'path/to/config.yaml' --resume 'xxxx/epoch-10.pkl'
```
Resuming from the latest snapshot file

```Bash
python3 ./main/dist_train_sat.py --config 'path/to/config.yaml' --resume latest
```