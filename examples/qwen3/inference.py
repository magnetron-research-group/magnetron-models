# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

# Chat with Qwen3 4B Instruct in the terminal, on CUDA if available.
# Usage: python -m examples.qwen3.inference [--cpu] [path/to/qwen3-4b-instruct-2507-bf16.mag]

from __future__ import annotations

from examples.chat import chat_arg_parser, load_llm, repl, select_device
from magnetron_models.utils import download_model


def main() -> None:
    args = chat_arg_parser('Chat with Qwen3 4B Instruct').parse_args()
    select_device(args.cpu)
    llm = load_llm(args.snapshot or download_model('mario-sieg/magnetron-qwen3', 'qwen3-4b-instruct-2507-bf16.mag'))
    repl(llm, args)


if __name__ == '__main__':
    main()
