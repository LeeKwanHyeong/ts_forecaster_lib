def test_public_api_import():
    import modeling_module as mm
    assert hasattr(mm, "train")
    assert hasattr(mm, "predict")
    assert hasattr(mm, "load_predictor")
    assert hasattr(mm, "build_dataloader")
    assert hasattr(mm, "build_dataset")