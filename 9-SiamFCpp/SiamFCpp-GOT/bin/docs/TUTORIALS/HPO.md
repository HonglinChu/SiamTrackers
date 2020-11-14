# Hyper-Parameter Search

Methods for video tasks usually contain a set of hyper-parameters and their combination of value can . Thus Hyper-Parameter Optimization (HPO).

Note that we only encourage HPO on validation subset of dataset. Any HPO on test subset will result in data leak by definition and such results will be "illegal" to publish.

## Usage

```bash
python3 main/hpo.py --config=experiments/siamfcpp/test/vot/siamfcpp_alexnet-multi_temp.yaml --hpo-config=experiments/siamfcpp/hpo/siamfcpp_SiamFCppMultiTempTracker-hpo.yaml
```

* _--config_ can be any experiment test
* _--hpo-config_ is the hpo config.
  * exp_save is specified in this hpo config .yaml file (by default _logs/hpo_).

### HPO configuration file

The .yaml file keeps the same structure as a test configuration, except that a suffix "_hpo_range" should be appended to the hyper-parameter to be searched.

```yaml
      SiamFCppMultiTempTracker:
        test_lr_hpo_range: [0.50, 0.58] # search between [0.50, 0.58]
        window_influence_hpo_range: [0.20, 0.22, 0.24, 0.26] # search from a set of values 0.20, 0.22, 0.24, 0.26
        mem_len_hpo_range: [3, 10] # search for integer between [3, 10)
        mem_len_hpo_range: [3.0, 10.0] # search for float between [3, 10)
```

## About HPO method

Currently, we use naive Random Search algorithm (e.g. [RandomSearch](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)) for its simplicity in implementation.

We are open to recommandation of other efficient and effective HPO methods. 
