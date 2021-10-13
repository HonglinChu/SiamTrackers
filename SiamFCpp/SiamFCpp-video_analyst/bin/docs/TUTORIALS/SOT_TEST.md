# SOT TEST 

## Usage of Test.py

Change directory to the repository root.

```Bash
python main/test.py --config experiments/siamfcpp/test/got10k/siamfcpp_alexnet-got.yaml
```

## Test Scripts

A collection of test scripts are located under _tools/test/_:

- [tools/test/test_VOT.sh](../../tools/siamfcpp/test/test_VOT.sh)
- [tools/test/test_GOT.sh](../../tools/siamfcpp/test/test_GOT.sh)
- [tools/test/test_LaSOT.sh](../../tools/siamfcpp/test/test_LaSOT.sh)
- [tools/test/test_OTB.sh](../../tools/siamfcpp/test/test_OTB.sh)

## Check test results

_EXP_NAME_ is the string value of key _test.track.exp_name_ in the corresponding _.yaml_ file.

### Check VOT results

```Bash
view logs/VOT2018/<EXP_NAME>.csv
```

### Check GOT-Benchmark results

GOT-Benchmark contains testers for a series of benchmarks, including OTB, VOT, LaSOT, GOT-10k, TrackingNet.

```Bash
view logs/GOT-Benchmark/report/GOT-10k/<EX015P_NAME>/performance.json
view logs/GOT-Benchmark/report/LaSOT/<EXP_NAME>/performance.json
view logs/GOT-Benchmark/report/otb2015/<EXP_NAME>/performance.json
view logs/GOT-Benchmark/report/TrackingNet/<EXP_NAME>/performance.json
```
