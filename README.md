# EchoFake: A Replay-Aware Dataset for Practical Speech Deepfake Detection

baseline models for EchoFake

Dataset is available at https://huggingface.co/datasets/EchoFake/EchoFake


To run models:
```bash
uv sync # install dependencies
# train rawnet2
python main.py --model rawnet2
# train aasist
python main.py --model rawnet2
# train wav2vec2
python main.py --model w2v

# eval models
python main.py --model rawnet2 --eval
```