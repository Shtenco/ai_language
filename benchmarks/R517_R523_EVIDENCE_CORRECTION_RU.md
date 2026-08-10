# NEXUS: корректировка доказательной базы R5.17–R5.23

Дата: 2026-08-10.

## 1. Критический дефект старого протокола

Фиксированный SentencePiece Unigram4096 использует `unk_id=0` и не имеет PAD (`pad_id=-1`). В R5.7–R5.16 значение `0` ошибочно использовалось как технический padding.

Это имело два следствия:

1. короткие free-generation prompts слева дополнялись десятками `<unk>`;
2. `A_PLAIN` в `prefix_ids()` получал 32 `<unk>` перед каждым контекстом, тогда как длинный `D_LOGIC_CYBER` prefix обычно занимал все 32 позиции реальными токенами.

Поэтому прежнее сравнение `D_LOGIC_CYBER > A_PLAIN` по обычному языковому BPB нельзя считать чистым доказательством пользы textual concept graph.

## 2. R5.17 PAD-clean frozen inference

Удаление left-UNK padding исправляет протокол, но само по себе не лечит frozen R5.12 32K cortex. На настоящих held-out 48-token contexts, где padding отсутствует изначально, greedy generation всё равно часто входит в локальные циклы.

Вывод: UNK-padding был серьёзным дефектом оценки, но autoregressive instability реальна независимо от него.

## 3. R5.18 variable-context retraining

Warm-start R5.12 32K, 4096 дополнительных шагов, тот же бюджет 2,998,620 параметров. Main context равномерно меняется от 4 до 48 токенов; left padding отсутствует; graph/retrieval context пересчитывается строго из тех же truncated context tokens; batch padding находится только справа после полного target.

BPB:

- SynTagRus: ctx8 `1.273632`, ctx16 `1.272200`, ctx32 `1.271377`, ctx48 `1.270398`;
- GSD: `1.649590`, `1.650012`, `1.649216`, `1.648105`;
- RuHeritage: `1.410839`, `1.405581`, `1.400908`, `1.399521`.

Variable-context curriculum делает качество почти нечувствительным к длине входного контекста. Однако raw greedy language остаётся нестабильным и часто циклическим.

## 4. R5.19 / R5.21 sequence critics

R5.19: 577-параметрический critic над статистиками target hidden states распознаёт грубые повторы, но почти не отличает настоящий continuation от настоящего continuation другого контекста.

R5.21: контекстно-целевой MLP critic (73,857 параметров) на train достигает `AUC(true > wrong-context)=0.9664`, но не переносится:

- SynTagRus: `0.5355`;
- GSD: `0.5473`;
- RuHeritage: `0.5963`.

Он по-прежнему хорошо распознаёт `repeat4` (`AUC≈0.87–0.89`), но это формальная, а не семантическая проверка.

Вывод: frozen 3M hidden states недостаточно хорошо поддерживают переносимую context-continuation compatibility для простого внешнего critic.

## 5. R5.20 prefix-rescue curve

На 64 held-out exact-48-token contexts модели давались первые `k=0/1/2/4/8/16/24/32` правильных future tokens, затем она отпускалась в greedy.

Даже при `k=32`:

- suffix exact ≈ `4.10%`;
- средняя правильная серия сразу после release ≈ `0.406` токена;
- вероятность первого правильного токена после release ≈ `25%`.

Следовательно, collapse не сводится к плохому старту последовательности. Ошибка возникает заново на протяжении всей траектории.

## 6. R5.23 TRUE vs SHUFFLED prefix dependence

Frozen R5.12 32K D-cortex. Primary causal contrast использует одинаковую конструкцию 32-token D-prefix; меняется только соответствие prefix содержанию контекста.

### Обычный русский

SynTagRus:

- TRUE `1.268018` BPB;
- SHUFFLED `1.268306`;
- FIXED `1.267946`;
- LEGACY_ZERO `1.395779`.

GSD:

- TRUE `1.699195`;
- SHUFFLED `1.699114`;
- FIXED `1.698946`;
- LEGACY_ZERO `1.887820`.

`TRUE ≈ SHUFFLED ≈ FIXED`, тогда как 32 `<unk>` резко ухудшают качество. Значит matching textual concept-prefix не имеет доказанного причинного языкового выигрыша в этой системе; прежнее `D>A` было существенно confounded плохим A-baseline.

### Logic

- TRUE `1.672942` BPB;
- SHUFFLED `1.731383`;
- delta TRUE-SHUFFLED = `-0.058442` BPB.

### Cyber

- TRUE `0.000202` BPB, top1 `99.9847%`;
- SHUFFLED `0.130188` BPB, top1 `95.5668%`;
- delta TRUE-SHUFFLED = `-0.129986` BPB.

Здесь matching exact state действительно причинно используется.

## 7. Новое архитектурное решение

Доказательства требуют разделить два органа:

```text
Unigram4096
    -> clean surface-language cortex

Exact / typed logic-cyber state
    -> отдельный reasoning-control organ
```

Textual concept graph больше не следует использовать как обязательный prefix обычного русского языка. Exact state сохраняется там, где TRUE-vs-SHUFFLED подтвердил причинную пользу.

Следующий основной тест — R5.24 matched capacity sweep 3M / ~8M / ~11.6M на clean surface LM: без textual graph-prefix, без left-UNK, с одинаковым corpus, tokenizer, source/target exposure и variable-context geometry.
