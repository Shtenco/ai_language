# AI Language Pro

AI Language Pro — это **рабочий прототип ИИ языка программирования** в Python.

Pipeline языка:

`instructions ↓ semantic graph ↓ AST ↓ Python / C / Rust / Solidity / Kotlin code → compiler`

Проект даёт реальный CLI и SDK, которые принимают `.ailang` инструкции и компилируют их в код целевого языка.

## Возможности

- Компиляция инструкций в:
  - Python
  - C
  - Rust
  - Solidity
  - Kotlin
- Построение **semantic graph**.
- Построение **AST**.
- Проверка Python-артефакта через bytecode compilation.
- Отдельный runtime режим `ask` для LLM-запросов с ключом пользователя.

## Установка

```bash
pip install ai-language-pro
```

или для разработки:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Формат инструкций (`.ailang`)

Построчный формат:

```text
ACTION TARGET | constraint1; constraint2
```

Пример:

```text
generate rest_api | auth; pagination
validate schema
emit docs | concise
```

## CLI

### 1) Сгенерировать код

```bash
ai-language generate examples/service.ailang --target python --out build/service.py --emit-graph build/graph.json
```

### 2) Проверить сгенерированный Python

```bash
ai-language check build/service.py
```

### 3) Использовать LLM runtime

```bash
export OPENAI_API_KEY="sk-..."
ai-language ask "Сгенерируй план миграции монолита в микросервисы" --model gpt-4o-mini --temperature 0.2
```

## Python SDK

```python
from ai_language import compile_source

source = """
generate payment_service | retries; idempotency
validate contracts
"""

result = compile_source(source, target="rust")
print(result.code)
print(len(result.semantic_graph.nodes), len(result.semantic_graph.edges))
```

## Запуск тестов

```bash
python -m ruff check .
PYTHONPATH=src pytest -q
```

## Публикация в PyPI

```bash
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

Если используете токен:

```bash
TWINE_USERNAME=__token__
TWINE_PASSWORD=<your-pypi-token>
python -m twine upload dist/*
```

## Страница пакета (PyPI)

- После успешной публикации пакет будет доступен по адресу:
  `https://pypi.org/project/ai-language-pro/`

## Безопасность

- Ключ OpenAI хранит и передаёт только пользователь.
- Не коммитьте `.env` и токены.

## Донаты

- **ETH**: `0x980Ddb04c54979b3Ed23df4a7DBc7049b7d0D686`
- **BTC**: `bc1q49rfm0p6qh6nlnm4az4yhhk9x82zfxwgtcnhvm`

## License

MIT
