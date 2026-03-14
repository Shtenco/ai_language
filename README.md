# AI Language Pro

## The Future of Software Delivery: **Instructions → Semantic Graph → AST → Code → Compiler → Machine**

**AI Language Pro** — коммерческий прототип нового поколения, который превращает естественные инструкции в исполняемые программные артефакты.

Это не просто "еще один AI-скрипт". Это **архитектура proto-agent систем**, где:

1. бизнес-интент задаётся как инструкции,
2. система строит семантическое представление,
3. формирует AST,
4. генерирует код в целевой язык,
5. компилирует/валидирует,
6. запускает на машине.

Если вы хотите продукт, который можно продавать enterprise-клиентам как AI automation platform, именно такая архитектура — будущее.

---

## Почему это продаёт вашу разработку

- **Понятный value for business:** от требований к рабочему коду и executable artifact.
- **Инженерная прозрачность:** semantic graph + AST позволяют объяснять решения и проводить аудит.
- **Мульти-языковой выход:** Python / C / Rust / Solidity / Kotlin.
- **Основа для proto-agent систем:** можно добавлять planner, memory, tool-execution без смены ядра.

---

## Что уже готово (рабочий Python-прототип)

- Парсер `.ailang` инструкций.
- Построение semantic graph.
- Построение AST.
- Генерация кода в 5 целевых языков.
- Валидация Python через bytecode compilation.
- Запуск Python-артефакта на машине (`run` команда).
- LLM runtime режим (`ask`) с пользовательским API-ключом.
- Тесты + линтинг + CI.

---

## Формат AI Language

```text
ACTION TARGET | constraint1; constraint2
```

Пример:

```text
generate payment_service | retries; idempotency
validate contracts
emit docs | concise
```

---

## CLI

### 1) Generate

```bash
ai-language generate examples/service.ailang --target python --out build/service.py --emit-graph build/graph.json
```

### 2) Check (compiler validation)

```bash
ai-language check build/service.py
```

### 3) Run (machine execution)

```bash
ai-language run build/service.py
```

### 4) Ask (LLM runtime)

```bash
export OPENAI_API_KEY="sk-..."
ai-language ask "Сгенерируй архитектуру сервиса антифрода" --model gpt-4o-mini --temperature 0.2
```

---

## Python SDK

```python
from ai_language import compile_source

source = """
generate anti_fraud_service | observability; retries
validate interfaces
"""

artifact = compile_source(source, target="rust")
print(artifact.code)
print(len(artifact.semantic_graph.nodes), len(artifact.semantic_graph.edges))
```

---

## Установка

```bash
pip install ai-language-pro
```

Для разработки:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

---

## Тесты

```bash
python -m ruff check .
PYTHONPATH=src pytest -q
```

---

## Коммерческая лицензия

Проект распространяется по коммерческой лицензии. Любое коммерческое использование, модификация,
встраивание в клиентские решения и перепродажа требуют отдельного письменного соглашения.

См. файл [LICENSE](LICENSE).

---

## Публикация в PyPI

```bash
python -m build
python -m twine check dist/*
TWINE_USERNAME=__token__ TWINE_PASSWORD=<your-pypi-token> python -m twine upload dist/*
```

После публикации пакет доступен по адресу:
`https://pypi.org/project/ai-language-pro/`

---

## Донаты

- **ETH:** `0x980Ddb04c54979b3Ed23df4a7DBc7049b7d0D686`
- **BTC:** `bc1q49rfm0p6qh6nlnm4az4yhhk9x82zfxwgtcnhvm`
