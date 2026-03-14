# AI Language Pro

[![CI](https://img.shields.io/badge/CI-pytest%20%7C%20ruff-blue)](#development)
[![Python](https://img.shields.io/badge/python-3.9%2B-success)](#requirements)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Профессиональный Python/PyPI-ready проект для генерации текста через LLM с удобным CLI, аккуратной архитектурой и безопасной моделью работы с ключом (ключ всегда указывает пользователь).

## ✨ Что внутри

- Современная структура `src/` + `pyproject.toml` (готово для публикации в PyPI).
- CLI-команда `ai-language`.
- Явное управление API ключом:
  - через аргумент `--api-key`, **или**
  - через переменную окружения `OPENAI_API_KEY` (включая `.env`).
- Базовые автотесты (`pytest`) и линтинг (`ruff`).
- GitHub Actions workflow для CI.

## 🧱 Структура проекта

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

## ✅ Requirements

- Python 3.9+
- OpenAI API key (пользователь задаёт самостоятельно)

## 🚀 Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

### Вариант 1: ключ через переменную окружения

```bash
export OPENAI_API_KEY="sk-..."
ai-language "Напиши краткий план запуска AI-продукта"
```

### Вариант 2: ключ напрямую в аргументе

```bash
ai-language "Сделай короткий питч" --api-key "sk-..."
```

### Выбор модели

```bash
ai-language "Summarize this text" --model gpt-4o-mini --temperature 0.3
```

## 🧪 Development

```bash
ruff check .
pytest
python -m build
```

## 📦 Публикация в PyPI

```bash
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

> Перед публикацией обновите `name`, `version`, `authors`, `urls` в `pyproject.toml`.

## 🔐 Безопасность API ключа

- Не коммитьте `.env` и реальные ключи.
- Для продакшена храните ключи в секретах CI/CD, Vault или cloud secret manager.

## 💖 Поддержать проект

- **ETH**: `0x980Ddb04c54979b3Ed23df4a7DBc7049b7d0D686`
- **BTC**: `bc1q49rfm0p6qh6nlnm4az4yhhk9x82zfxwgtcnhvm`

## 📄 License

MIT
