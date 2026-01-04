from modeling_module import train, load_predictor

# train
result = train({
    "seed": 42,
    "model": {"name": "PatchMixer_Base", "d_model": 128},
    "data": {"path": "/path/to/dataset.parquet"},
    "trainer": {"epochs": 50, "lr": 3e-4},
})

# infer
predictor = load_predictor(result.best_ckpt_path)
yhat = predictor(batch)