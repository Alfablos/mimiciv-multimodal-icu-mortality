# MMIM Orchestrator

Example test materialization config:

```yaml
ops:
  dataset_manifest:
    config:
      debug: false
      max_workers: 4
      output_dir: ./out
      manifest_uri: file://out/manifest.json
  training_run:
    config:
      batch_size: 32
      dropout: 0.3
      epochs: 1
      learning_rate: 0.0001
      # Portion of training/validation sets to use. Remember, if too low val_loss will be `NaN`!
      train_limit: 0.01
      working_directory: ./out
  quality_gate:
    config:
      AUPRC: 0.5
      AUROC: 0.7
      sens_at_95_spec: 0.7
      # set fake_pass to true to fake a better model than static and dynamic quality gates or lower the metric corresponding to model_selection_metric if the model performs bad (testing only)
      fake_pass: true
      model_selection_metric: AUROC
resources: {}
```
