# Baybayin OCR

Baybayin text-line OCR using PyTorch, a compact CNN/TCN, and CTC loss.

## Setup

1. Create the environment:

    ```bash
    python3 -m venv .venv
    ```

2. Activate the environment:

    ```bash
    # For Windows
    .venv\Scripts\activate

    # For Linux
    source .venv/bin/activate
    ```

3. Install the dependencies:

    ```bash
    python3 -m pip install -r requirements.txt
    ```

## CUDA Setup on Windows or Linux

Windows PowerShell:

```powershell
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

Linux:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python3 -m pip install -r requirements.txt
```

## Dataset

Put single-line or text-region images in `dataset/images/`. Create `dataset/labels.txt` as UTF-8, with one filename and label per line separated by a literal tab:

```text
img_001.png<TAB>ᜀᜆᜄᜎ
img_002.png<TAB>ᜊᜎᜈᜌᜒ
```

## Train

```bash
python3 train.py --dataset dataset --epochs 60 --batch-size 64 --image-width 320
```

Training opens a resizable live graph window. Use `--no-live-plot` to disable it. Checkpoints, metrics, and graphs are saved in `checkpoints/` by default.

Other commands:

```bash
python3 train.py --resume checkpoints/last_model.pt
python3 predict.py --checkpoint checkpoints/best_model.pt --image dataset/images/img_001.png
python3 -m pytest -q tests
```

## Benchmark

1. Train a model and choose a representative single-line image.
2. Run the benchmark (defaults: 20 warmups, 100 iterations, JSON saved to `runs/benchmark.json`):

    ```bash
    python3 benchmark.py --checkpoint checkpoints/best_model.pt --image dataset/images/img_00001.png
    ```

Use `--device cuda` or set `--threads` for CPU runs. Compare model-only and full preprocessing/inference timings on the target device.

For an exported LiteRT model:

```bash
python3 benchmark_litert.py --model checkpoints/baybayin_mobile.tflite --checkpoint checkpoints/best_model.pt --image dataset/images/img_00001.png
```

## LiteRT export

Use Linux with Python 3.11 for the converter environment:

1. Create an environment for model export:

    ```bash
    python3.11 -m venv .venv-litert
    ```

2. Activate the environment:

    ```bash
    source .venv-litert/bin/activate
    ```

3. Install the environment:

    ```bash
    python3 -m pip install -r requirements.txt
    ```

4. Run export command:

    ```bash
    python3 export_litert.py --checkpoint checkpoints/best_model.pt --output checkpoints/baybayin_mobile.tflite
    ```

## Model input

Default input shape is `(B, 1, 48, 320)`. Images are grayscale, fit to a white canvas, and normalized to `[-1, 1]`. Keep dimensions fixed for each checkpoint and export. Check accuracy on held-out, camera-like data and measure latency on the target device.

## License

This project is licensed under the terms described in the [LICENSE](./LICENSE) file.
