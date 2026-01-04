# andgein.ru 2026

Solutions for [andgein.ru 2026](https://2026.andgein.ru) programming contest.

## Tasks

| Task | Description |
|------|-------------|
| [lego](lego/) | Count LEGO minifigures in images using Grounding DINO + CLIP ([detailed report](lego/lego.md)) |
| [uneval](uneval/) | Find expressions with specific constraints |
| [brainfuck](brainfuck/) | Generate Brainfuck programs that output target strings |
| [bulls_and_cows](bulls_and_cows/) | Bulls and Cows solver using Z3 SMT (C# + Z3Wrap, ~3s) |

## Setup

```bash
uv sync
cp .env.example .env  # Add your API key
```

## Usage

Python tasks:
```bash
uv run python <task>/<task>.py
```

C# file-based apps (requires .NET 10+):
```bash
dotnet run <task>/<task>.cs
```
