from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generative_arm_text.data import ByteTokenizer
from generative_arm_text.model import GenerativeARMConfig, GenerativeARMLanguageModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with a trained Generative ARM model.")
    parser.add_argument("--checkpoint", type=str, default=str(HERE / "runs" / "tiny_arm_lm" / "best.pt"))
    parser.add_argument("--prompt", type=str, default="Algebraic")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = torch.load(Path(args.checkpoint), map_location=args.device)
    config = GenerativeARMConfig.from_dict(checkpoint["config"])
    model = GenerativeARMLanguageModel(config).to(args.device)
    model.load_state_dict(checkpoint["model"])

    tokenizer = ByteTokenizer()
    prompt_ids = torch.tensor([tokenizer.encode(args.prompt, add_bos=True)], dtype=torch.long, device=args.device)
    output_ids = model.generate(
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_id=tokenizer.eos_id,
    )[0].tolist()
    print(tokenizer.decode(output_ids))


if __name__ == "__main__":
    main()
