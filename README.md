# andgein.ru 2026

Solutions for [andgein.ru 2026](https://2026.andgein.ru) programming contest.

## Tasks

| Task | Description |
|------|-------------|
| [lego](lego/) | Count LEGO minifigures in images using Grounding DINO + CLIP ([detailed report](lego/lego.md)) |
| [uneval](uneval/) | Find expressions with specific constraints |
| [brainfuck](brainfuck/) | Generate Brainfuck programs that output target strings |

## Setup

```bash
uv sync
cp .env.example .env  # Add your API key
```

## Usage

```bash
uv run python <task>/<task>.py
```
