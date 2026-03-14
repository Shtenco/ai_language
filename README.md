# AI Language Pro — ИИ язык программирования (Python SDK + CLI)

[![CI](https://img.shields.io/badge/CI-pytest%20%7C%20ruff-blue)](#development)
[![Python](https://img.shields.io/badge/python-3.9%2B-success)](#requirements)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**AI Language Pro** — это проект, который подаёт себя как **ИИ язык программирования нового поколения**:
вы пишете естественные инструкции как код, а рантайм-компоненты (`CLI`/`SDK`) превращают их в управляемые вызовы LLM.

> Идея: естественный язык как high-level DSL для программирования ИИ-поведения.

---

## Что такое «ИИ язык программирования» в этом проекте

В контексте `ai-language-pro` язык состоит из трёх уровней:

1. **Syntax Layer (Prompt-as-Code)**
   - Текстовые инструкции выступают как исходный код.
2. **Runtime Layer (CLI/SDK)**
   - `ai-language` CLI и Python-клиент исполняют этот «код» через API модели.
3. **Execution Layer (LLM Model)**
   - Модель (`gpt-4o-mini` по умолчанию) генерирует результат.

Такой подход позволяет использовать проект как основу для:
- AI-скриптинга,
- автоматизации текстовых пайплайнов,
- создания proto-agent систем.

---

## Пакет в PyPI

- **Имя пакета:** `ai-language-pro`
- **Страница пакета (PyPI):** `https://pypi.org/project/ai-language-pro/`

> Если страница ещё не отображается сразу, подождите несколько минут: индекс PyPI обновляется не мгновенно.

---

## Архитектура

```text
.
├── pyproject.toml
├── src/
│   └── ai_language/
│       ├── __init__.py
│       ├── cli.py
│       ├── client.py
│       └── config.py
├── tests/
│   ├── test_cli.py
│   └── test_config.py
└── .github/workflows/ci.yml
```

### Основные компоненты

- `config.py` — безопасное получение ключа пользователя (`--api-key` или `OPENAI_API_KEY`).
- `client.py` — Python runtime-клиент (`AILanguageClient`) для генерации текста.
- `cli.py` — исполняемый вход в «язык» через команду `ai-language`.

---

## Requirements

- Python 3.9+
- OpenAI API key (пользователь указывает самостоятельно)

---

## Установка

### Из PyPI

```bash
pip install ai-language-pro
```

### Для разработки

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

---

## Быстрый старт (как писать на AI Language)

### 1) Через переменную окружения

```bash
export OPENAI_API_KEY="sk-..."
ai-language "Сгенерируй архитектуру микросервиса для платежей"
```

### 2) Через аргумент

```bash
ai-language "Сделай техспеку API" --api-key "sk-..."
```

### 3) Управление моделью/температурой

```bash
ai-language "Explain monads simply" --model gpt-4o-mini --temperature 0.3
```

---

## Python SDK

```python
from ai_language.client import AILanguageClient

client = AILanguageClient(api_key="sk-...", model="gpt-4o-mini")
result = client.generate("Сгенерируй шаблон RFC для команды backend")
print(result)
```

---

## Development

```bash
ruff check .
PYTHONPATH=src pytest -q
python -m build
```

---

## Публикация новой версии в PyPI

```bash
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

Перед публикацией обновите `version` в `pyproject.toml`.

---

## Безопасность

- Не храните реальные API ключи в Git.
- Не коммитьте `.env`.
- Для продакшена используйте secrets manager.

---

## Донаты

- **ETH:** `0x980Ddb04c54979b3Ed23df4a7DBc7049b7d0D686`
- **BTC:** `bc1q49rfm0p6qh6nlnm4az4yhhk9x82zfxwgtcnhvm`

---

## License

MIT
