def test_public_api_import():
    import modeling_module as mm
    assert hasattr(mm, "train")
    assert hasattr(mm, "predict")
    assert hasattr(mm, "load_predictor")
    assert hasattr(mm, "build_dataloader")
    assert hasattr(mm, "build_dataset")
    assert hasattr(mm, "forecast")
    assert hasattr(mm, "ForecastRequest")
    assert hasattr(mm, "ForecastRuntimeConfig")
    assert hasattr(mm, "ForecastResult")


def test_api_subpackage_exports_public_surface():
    import modeling_module.api as api

    assert hasattr(api, "train")
    assert hasattr(api, "predict")
    assert hasattr(api, "load_predictor")
    assert hasattr(api, "build_dataloader")
    assert hasattr(api, "build_dataset")
    assert hasattr(api, "TrainRequest")
    assert hasattr(api, "DataRequest")
    assert hasattr(api, "forecast")
    assert hasattr(api, "ForecastRequest")
    assert hasattr(api, "ForecastRuntimeConfig")
    assert hasattr(api, "ForecastResult")
