# VOS TEST 

A collection of test scripts are located under _tools/test/_:

- [tools/test/test_DAVIS2017.sh](../tools/test/test_DAVIS2017.sh)


## Check test results

_EXP_NAME_ is the string value of key _test.vos.exp_name_ in the corresponding _.yaml_ file.

### Check DAVIS global results 

```Bash
cat logs/DAVIS2017/EXP_NAME/baseline/dump/default_hp_global_results.csv
```

### Check DAVIS results for each sequence

```Bash
cat logs/DAVIS2017/EXP_NAME/baseline/dump/default_hp_name_per_sequence_results.csv
```
