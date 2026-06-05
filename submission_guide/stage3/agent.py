import json
from pathlib import Path


def main():
    model_config_path = Path("target/model_config.json")
    weight_path = Path("target/weights/model.pt")
    engine_path = Path("workspace/engine.py")

    with model_config_path.open() as f:
        model_config = json.load(f)

    if not weight_path.exists():
        raise FileNotFoundError(f"missing weight file: {weight_path}")
    if not engine_path.exists():
        raise FileNotFoundError(f"missing engine implementation: {engine_path}")

    print("我什么都没做，代码已经在 workspace/engine.py 里了。")
    print(f"Existing engine implementation: {engine_path}")
    print(f"Read model config: {model_config_path}")
    print(f"Model type: {model_config.get('model_type')}")
    print(f"Layers: {model_config.get('num_hidden_layers')}")
    print(f"Hidden size: {model_config.get('hidden_size')}")
    print(f"Weight file: {weight_path}")


if __name__ == "__main__":
    main()
