"""The gesture models we compare (kept free of torch imports so the UI can list them instantly).

"kind" selects the input: "seq" = landmark sequences, "clip" = hand-crop clips.
"""
MODEL_SPECS = {
    "mlp": {"kind": "seq", "arch": "mlp", "motion": True, "label": "MLP single-frame (baseline)"},
    "lstm": {"kind": "seq", "arch": "lstm", "motion": True, "label": "LSTM on landmarks (main)"},
    "gru": {"kind": "seq", "arch": "gru", "motion": True, "label": "GRU on landmarks"},
    "lstm_shape": {"kind": "seq", "arch": "lstm", "motion": False,
                   "label": "LSTM without motion features (ablation)"},
    "cnn": {"kind": "clip", "arch": "cnn_gru", "motion": False, "label": "CNN+GRU on hand crops"},
}
