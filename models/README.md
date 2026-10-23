# Model Weights

After running `python train.py`, this directory will contain:

- `best_model.pth`         — Best ResNet-18 checkpoint (by validation accuracy)
- `training_history.json`  — Per-epoch loss and accuracy log
- `evaluation_report.txt`  — Classification report on test set
- `class_mapping.json`     — Index-to-class name mapping
- `eval_summary.json`      — Per-class precision/recall/F1 (after running evaluate.py)

The Flask server loads `best_model.pth` automatically on startup.
If the file is missing, inference returns a 503 with a clear error message.
