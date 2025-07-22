import torch
from transformers.pipelines import pipeline


def test_slm():
    pl = pipeline(model="openai-community/gpt2", torch_dtype=torch.float16, device=0)

    print(pl("Quelle est la capitale de la France ?"))


if __name__ == "__main__":
    test_slm()
