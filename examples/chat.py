# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

import argparse
import contextlib
import re
import time
from collections.abc import Callable
from pathlib import Path

with contextlib.suppress(ImportError):
    import readline

from magnetron import context
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from magnetron_models.inference import InferenceAdapter, load
from magnetron_models.inference.adapters import StopReason, TextChunk, TextRequest

console = Console()

_COMMANDS: list[tuple[str, str]] = [
    ('/reset', 'forget the conversation'),
    ('/system <text>', 'set the system prompt (and reset)'),
    ('/think', 'toggle thinking mode (if the model supports it)'),
    ('/stats', 'toggle tokens/s output after each reply'),
    ('/exit', 'quit (or ctrl+d)'),
    ('ctrl+c', 'stop the current reply'),
]
_THINK_OPEN: str = '<think>'
_THINK_CLOSE: str = '</think>'
_FENCE = re.compile(r'^\s*(```|~~~)', re.MULTILINE)
_THINK_STYLE: str = 'dim italic'


def chat_arg_parser(description: str, *, cuda: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('snapshot', nargs='?', type=Path, default=None, help='Local .mag snapshot, downloaded from HF if omitted')
    if cuda:
        parser.add_argument('--cpu', action='store_true', help='Run on the CPU even if CUDA is available')
    parser.add_argument('--system', type=str, default='You are a helpful assistant.', help='System prompt')
    parser.add_argument('--max-toks', type=int, default=2048, help='Max tokens per reply')
    parser.add_argument('--temp', type=float, default=0.7, help='Sampling temperature, 0 = greedy')
    parser.add_argument('--topk', type=int, default=20, help='Top-k sampling, 0 = disabled')
    return parser


def select_device(cpu: bool) -> str:
    device = 'cuda:0' if not cpu and context.is_device_available('cuda:0') else 'cpu'
    context.set_default_device(device)
    return device


class MarkdownStream:
    def __init__(self, live: Live, thinking: bool) -> None:
        self.live = live
        self.thinking = thinking
        self.pending = ''
        self.started = False

    def feed(self, delta: str) -> None:
        self.pending += delta
        if not self.started:
            stripped = self.pending.lstrip()
            if len(stripped) < len(_THINK_OPEN) and _THINK_OPEN.startswith(stripped):
                return
            self.started = True
            if stripped.startswith(_THINK_OPEN):
                self.thinking, self.pending = True, stripped[len(_THINK_OPEN) :]
        if self.thinking and _THINK_CLOSE in self.pending:
            reasoning, self.pending = self.pending.split(_THINK_CLOSE, 1)
            self._commit(Text(reasoning.strip(), style=_THINK_STYLE))
            self._commit(Rule(style='dim'))
            self.thinking = False
            self.pending = self.pending.lstrip('\n')
        if self.thinking:
            if (cut := self.pending.rfind('\n')) != -1:
                self._commit(Text(self.pending[:cut], style=_THINK_STYLE))
                self.pending = self.pending[cut + 1 :]
        elif (cut := self._safe_markdown_cut(self.pending)) > 0:
            self._commit(Markdown(self.pending[:cut]))
            self.pending = self.pending[cut:].lstrip('\n')
        self.live.update(self._tail(), refresh=False)

    def close(self) -> None:
        self.live.update(Text(''), refresh=True)
        if self.pending.strip():
            self._commit(self._tail())
        self.pending = ''

    def _tail(self) -> RenderableType:
        if self.thinking:
            return Text(self.pending, style=_THINK_STYLE)
        return Markdown(self.pending)

    def _commit(self, r: RenderableType) -> None:
        if not isinstance(r, Text) or r.plain:
            self.live.console.print(r)

    @staticmethod
    def _safe_markdown_cut(text: str) -> int:
        cut = text.rfind('\n\n')
        while cut > 0:
            if len(_FENCE.findall(text, 0, cut)) % 2 == 0:
                return cut
            cut = text.rfind('\n\n', 0, cut)
        return -1


def _print_help() -> None:
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style='bold cyan')
    table.add_column(style='dim')
    for cmd, desc in _COMMANDS:
        table.add_row(cmd, desc)
    console.print(table)
    console.print()


def _info(msg: str) -> None:
    console.print(f'[dim]\\[{msg}][/dim]')


def repl(llm: InferenceAdapter, args: argparse.Namespace, *, on_think: Callable[[bool], None] | None = None, think: bool = False) -> None:
    system: str = args.system
    history: list[dict[str, str]] = []
    stats = False
    if on_think is not None:
        on_think(think)
    _print_help()
    while True:
        try:
            line = console.input('[bold green]> [/]').strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if not line:
            continue
        if line.startswith('/'):
            cmd, _, rest = line.partition(' ')
            match cmd:
                case '/exit' | '/quit':
                    return
                case '/reset':
                    history.clear()
                    _info('conversation cleared')
                case '/system':
                    system = rest.strip()
                    history.clear()
                    _info(f'system prompt set, conversation cleared: {system!r}')
                case '/think':
                    if on_think is None:
                        _info('this model has no thinking mode')
                    else:
                        think = not think
                        on_think(think)
                        _info(f'thinking {"on" if think else "off"}')
                case '/stats':
                    stats = not stats
                    _info(f'stats {"on" if stats else "off"}')
                case _:
                    _print_help()
            continue

        history.append({'role': 'user', 'content': line})
        msgs = ([{'role': 'system', 'content': system}] if system else []) + history
        req = TextRequest(prompt=msgs, max_toks=args.max_toks, temp=args.temp, topk=args.topk)
        reply: list[str] = []
        ntoks, reason = 0, StopReason.STOP
        t0 = time.perf_counter()
        console.print()
        chunk: TextChunk
        with Live(Group(), console=console, refresh_per_second=15, transient=False) as live:
            out = MarkdownStream(live, thinking=think and on_think is not None)
            try:
                for chunk in llm.stream(req):
                    ntoks += chunk.token is not None
                    reply.append(chunk.delta)
                    out.feed(chunk.delta)
                    if chunk.stop_reason is not None:
                        reason = chunk.stop_reason
            except KeyboardInterrupt:
                reason = StopReason.CANCELLED
            finally:
                out.close()
        dt = time.perf_counter() - t0
        if reason is not StopReason.STOP:
            _info(reason.value)
        if stats:
            _info(f'{ntoks} tokens in {dt:.1f} s, {ntoks / dt:.1f} tok/s incl. prefill')
        console.print()
        history.append({'role': 'assistant', 'content': ''.join(reply)})


def load_llm(snapshot: Path) -> InferenceAdapter:
    with console.status(f'loading [bold]{snapshot.name}[/] on {context.get_default_device()}...'):
        t0 = time.perf_counter()
        llm = load(snapshot)
    _info(f'loaded {snapshot.name} in {time.perf_counter() - t0:.1f} s')
    console.print()
    return llm
