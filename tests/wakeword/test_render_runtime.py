from wakeword.render_runtime import render_openwakeword_execstart


def test_execstart_contains_model_name_and_custom_dir():
    line = render_openwakeword_execstart(
        run_script="/home/samda/wyoming-openwakeword/script/run",
        custom_model_dir="/home/samda/wyoming-openwakeword/custom-models",
        model_name="igor",
    )
    assert "/home/samda/wyoming-openwakeword/script/run" in line
    assert "--custom-model-dir /home/samda/wyoming-openwakeword/custom-models" in line
    assert "--preload-model igor" in line
    assert "--threshold 0.5" in line
    assert "--trigger-level 3" in line
