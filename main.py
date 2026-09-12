import torch
import transformers


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    device = get_device()
    print(f"torch        {torch.__version__}")
    print(f"transformers {transformers.__version__}")
    print(f"device       {device}")

    # Tiny tensor op to confirm the compute backend works.
    x = torch.randn(2, 3, device=device)
    print(f"sample tensor mean: {x.mean().item():.4f}")


if __name__ == "__main__":
    main()
