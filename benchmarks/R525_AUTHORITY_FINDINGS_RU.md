# R5.25 — Exact Authority Bus

## Что исправлено

После аудита R5.7–R5.23 exact logic/cyber больше не считается задачей вероятностного surface LM. Добавлен `nexus/authority_bus.py`:

- узкая маршрутизация только при точном совпадении с поддерживаемой grammar;
- forward-chaining BFS для логических правил;
- явный proof path и proof depth;
- exact feedback-error controller для cyber-задач;
- SHA-256 provenance входа;
- canonical Russian response;
- обычный русский, вопросы и свободный текст не перехватываются и уходят в surface cortex.

## Самопроверка

На фиксированном seed выполнено по 50 000 случайных exact случаев каждого класса:

- logic: `50 000 / 50 000 = 100%`;
- cyber: `50 000 / 50 000 = 100%`;
- обычные тексты: `4 / 4` корректно ушли в fallthrough.

## Найденный дефект старого logic benchmark

В legacy `make_logic_example()` для отрицательного класса использовалось:

```python
fact = rng.choice([x for x in pool if x not in (p1, p2)])
claim = p3
```

То есть `p3` не исключался из возможного `fact`. Когда `fact == p3`, вход прямо утверждает claim как истинный факт, но legacy target всё равно говорит «Нет».

Аналитическая вероятность противоречия:

`P(negative) * P(fact=p3 | negative) = 0.35 * 1/13 = 0.026923...`

Monte Carlo audit на 200 000 legacy examples:

- negative examples: 70 048;
- contradictory labels: 5 440;
- contradiction rate: **2.72%**.

Это совпадает с аналитическим ожиданием.

## Следствие для прежних результатов

Logic-метрики R5.7–R5.23 содержат примерно 2.7% внутренне противоречивых labels и не должны использоваться как точный benchmark формальной логики. Cyber benchmark такого найденного дефекта не имеет и остаётся более чистой проверкой exact-state dependence.

Начиная с R5.25, truth для exact logic определяется только достижимостью claim из фактов по правилам, а не заранее выбранным random label.
