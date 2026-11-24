# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

from examples.chat import chat_arg_parser, load_llm, repl, select_device
from magnetron_models.models.qwen3hyb.inference import Qwen3HybInferenceAdapter
from magnetron_models.utils import download_model


def main() -> None:
    parser = chat_arg_parser('Chat with Qwen 3.8 27B on the CPU', cuda=False)
    parser.add_argument('--think', action='store_true', help='Start with thinking mode on')
    args = parser.parse_args()
    select_device(cpu=True)
    llm = load_llm(args.snapshot or download_model('mario-sieg/magnetron-qwen3.8', 'qwen3.8-27b-bf16.mag'))
    assert isinstance(llm, Qwen3HybInferenceAdapter), 'not a qwen3hyb snapshot'
    repl(llm, args, on_think=lambda on: setattr(llm, 'enable_thinking', on), think=args.think)


if __name__ == '__main__':
    main()
