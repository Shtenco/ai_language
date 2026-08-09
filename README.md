# AI Language Pro

**AI Language Pro** объединяет два слоя:

1. LLM-meta язык: `instructions → semantic graph → AST → code → compiler → machine`.
2. Локальный **AI coding agent** для работы с реальными репозиториями из терминала.

Никакого Node.js и `npm install`: CLI написан на Python и упакован для `pip` / `pipx`.

## AI coding agent

После установки пакет добавляет короткую команду `ail`.

```bash
pipx install ai-language-pro
```

или:

```bash
python -m pip install ai-language-pro
```

Задайте ключ OpenAI:

```bash
export OPENAI_API_KEY="sk-..."
```

PowerShell:

```powershell
$env:OPENAI_API_KEY="sk-..."
```

Перейдите в любой Git-репозиторий и запустите:

```bash
ail
```

Получится интерактивная сессия:

```text
AI Language Agent | model=gpt-5.6 | workspace=/path/to/project
Commands: /help, /clear, /diff, /exit
ail>
```

Можно дать задачу одной командой:

```bash
ail "Найди причину падения тестов, исправь код и снова запусти тесты"
```

По умолчанию агент просит подтверждение перед изменением файлов и запуском локальных команд. Для автономного режима:

```bash
ail -y "Проведи рефакторинг parser, обнови тесты и проверь весь test suite"
```

Только анализ без изменений:

```bash
ail --read-only "Проведи архитектурный аудит проекта"
```

Выбор модели и reasoning effort:

```bash
ail --model gpt-5.6 --reasoning high "Оптимизируй критический путь"
```

Эквивалентная полная команда:

```bash
ai-language agent "Исправь баг"
```

## Что умеет агент

- рекурсивно просматривать структуру репозитория;
- читать файлы с номерами строк;
- искать текст по проекту;
- создавать и переписывать файлы;
- выполнять точечные exact-text edits;
- удалять отдельные файлы;
- запускать тесты, линтеры, компиляторы и другие прямые локальные команды;
- смотреть `git status` и `git diff`;
- продолжать многошаговую задачу через Responses API tool-calling loop;
- хранить контекст интерактивной сессии между запросами.

### Локальные границы безопасности

Agent tools привязаны к `--cwd` и не позволяют читать/писать пути за пределами workspace. `.env`, ключевые credential-файлы, `.ssh`, `.aws` и похожие secret locations скрываются от file tools. Привилегированные и явно разрушительные команды, shell chaining и mutating Git-команды блокируются.

Для максимальной изоляции используйте:

```bash
ail --read-only
```

или запускайте агента в отдельном контейнере/VM.

## Команды интерактивного режима

```text
/help   краткая справка
/clear  сброс контекста модели
/diff   git status + git diff
/exit   выход
```

## AI Language compiler

Существующий LLM-meta язык остаётся совместимым.

Формат `.ailang`:

```text
ACTION TARGET | constraint1; constraint2
```

Пример:

```text
generate payment_service | retries; idempotency
validate contracts
emit docs | concise
```

Генерация Python:

```bash
ai-language generate examples/service.ailang \
  --target python \
  --out build/service.py \
  --emit-graph build/graph.json
```

Поддерживаемые targets:

- Python
- C
- Rust
- Solidity
- Kotlin

Проверка сгенерированного Python:

```bash
ai-language check build/service.py
```

Запуск:

```bash
ai-language run build/service.py
```

Одиночный LLM request:

```bash
ai-language ask "Спроектируй anti-fraud сервис"
```

## Python SDK

```python
from ai_language import CodingAgent, Workspace, compile_source

artifact = compile_source(
    "generate anti_fraud_service | observability; retries",
    target="rust",
)
print(artifact.code)
```

Agent primitives тоже доступны из Python:

```python
from pathlib import Path

from ai_language import CodingAgent, Workspace

workspace = Workspace(Path.cwd(), auto_approve=False)
agent = CodingAgent(workspace=workspace, model="gpt-5.6")
print(agent.run("Проанализируй архитектуру проекта"))
```

## Разработка

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
ruff check .
pytest
python -m build
```

## Публикация в PyPI

Репозиторий содержит `.github/workflows/publish.yml`. Workflow запускается при публикации GitHub Release, собирает wheel/sdist и отправляет их в PyPI через **Trusted Publishing (OIDC)** без хранения PyPI API token в GitHub secrets.

Один раз в настройках проекта `ai-language-pro` на PyPI нужно добавить trusted publisher для:

```text
Owner: Shtenco
Repository: ai_language
Workflow: publish.yml
Environment: pypi
```

После этого обычный релиз GitHub публикует новую версию автоматически.

## Лицензия

Проект распространяется по коммерческой лицензии. Подробности: [LICENSE](LICENSE).

## Донаты

- **ETH:** `0x980Ddb04c54979b3Ed23df4a7DBc7049b7d0D686`
- **BTC:** `bc1q49rfm0p6qh6nlnm4az4yhhk9x82zfxwgtcnhvm`
