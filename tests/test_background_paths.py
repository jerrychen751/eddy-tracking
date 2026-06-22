from background import background_output_dir


def test_background_output_dir_lives_under_pigments_background():
    experiment = "gulf_stream_pigment_influencers_20241001_20251231"

    path = background_output_dir(experiment)

    assert path.parts[-5:] == ("data", experiment, "silver", "pigments", "background")
    assert path.name == "background"
