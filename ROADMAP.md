# Roadmap

1. ~~Basic orchestration~~
2. Baseline model (Decision Tree/Linear Regression and CNN with binary classification) for standalone tabular-only and visual-only predictions.
3. Full orchestration:
    1. Orchestration also compares to baseline model
    2. Model deploy (platform: to be defined)
    3. Model test (test set, shadow testing, Kubernetes + traffic mirroring?)
    4. Revert to previous model if performance on the test set is not better than that on the validation test, log situation
4. Monitoring
5. Dashbord + explanatory LLM (monitoring system connector) + trusted frontend components
6. GPC/AWS training
70. Federated training
