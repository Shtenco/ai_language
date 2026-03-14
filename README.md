# AI Language Pro

## 🌌 Новая эпоха программирования: когда мысль становится кодом

Представьте, что вы больше не «пишете код» в привычном смысле.
Вы формулируете **намерение** простыми словами — а система превращает его в исполняемую программу.

**AI Language Pro** — это LLM-meta язык программирования и компиляции.
Он строит мост между человеческой идеей и машинным выполнением:

`instructions → semantic graph → AST → Python/C/Rust/Solidity/Kotlin code → compiler → machine`

Это не «замена Python». Это уровень **выше** обычных языков — слой архитектурного управления программами.

---

## 🚀 Почему это принципиально круче обычных языков

Обычные языки отвечают на вопрос: **«как именно реализовать?»**  
AI Language Pro сначала отвечает на вопрос: **«что именно должно быть сделано и зачем?»**

### Что это даёт бизнесу и разработке

- **Скорость x10 на старте идей** — от инструкции к рабочему артефакту за минуты.
- **Прозрачность решений** — semantic graph и AST показывают, как мысль превращается в программу.
- **Мульти-платформенность с одного источника** — один intent, несколько языков вывода.
- **Архитектурная масштабируемость** — идеальная основа для proto-agent систем нового поколения.
- **Переход от «файлов кода» к «компилируемому смыслу»** — это и есть будущее программирования.

---

## 🧠 Что такое LLM-meta язык простыми словами

LLM-meta язык — это язык, который программирует **не строки**, а **смысл**:

1. Вы задаёте инструкцию простым человеческим языком.
2. Система строит карту смысла (semantic graph).
3. Преобразует её в структурное представление (AST).
4. Генерирует код под выбранную технологию.
5. Проверяет/компилирует.
6. Запускает на машине.

В обычном мире вы думаете как компилятор.
В AI Language Pro компилятор начинает думать как вы.

---

## ✅ Что уже реализовано (рабочий прототип на Python)

- Парсер `.ailang` инструкций.
- Построение semantic graph.
- Построение AST.
- Генерация кода в 5 языков: Python, C, Rust, Solidity, Kotlin.
- Валидация Python-кода через bytecode compile.
- Машинный запуск Python-артефакта (`run`).
- LLM runtime (`ask`) с ключом пользователя.
- Тесты + линтинг + CI.

---

## ✍️ Формат AI Language

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

## 🛠 CLI

### 1) Генерация кода

```bash
ai-language generate examples/service.ailang --target python --out build/service.py --emit-graph build/graph.json
```

### 2) Проверка (compiler validation)

```bash
ai-language check build/service.py
```

### 3) Запуск на машине (machine execution)

```bash
ai-language run build/service.py
```

### 4) LLM runtime

```bash
export OPENAI_API_KEY="sk-..."
ai-language ask "Спроектируй anti-fraud сервис для финтеха" --model gpt-4o-mini --temperature 0.2
```

---

## 🐍 Python SDK

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

## ⚙️ Локальный запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

---

## 🧪 Тесты

```bash
python -m ruff check .
PYTHONPATH=src pytest -q
```

---

## 💼 Коммерческая лицензия

Проект распространяется по коммерческой лицензии.
Коммерческое внедрение, кастомизация и интеграция в клиентские системы — по отдельному соглашению.

Подробности: [LICENSE](LICENSE)

---

## 💎 Донаты

- **ETH:** `0x980Ddb04c54979b3Ed23df4a7DBc7049b7d0D686`
- **BTC:** `bc1q49rfm0p6qh6nlnm4az4yhhk9x82zfxwgtcnhvm`
