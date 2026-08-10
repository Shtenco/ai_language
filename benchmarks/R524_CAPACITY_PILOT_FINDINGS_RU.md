# R5.24 Capacity Pilot — чистый surface-cortex scaling

Дата: 2026-08-10.

## Протокол

Это первый NEXUS capacity comparison после исправления старых confounds:

- фиксированный lossless Unigram4096;
- textual graph-prefix отсутствует;
- surface-prefix отсутствует;
- left padding отсутствует;
- variable context 4..48;
- target = 48 токенов;
- только документы, содержащие минимум 96 tokenizer tokens;
- 1024 optimizer steps, batch 8;
- **393,216 target tokens / 2,217,299 raw target bytes для каждой модели**;
- один и тот же `DATA_SEED=20261334`;
- одни и те же training windows в одном порядке для 3M / 8M / 11M;
- одинаковый LR = 5e-4.

Сравнивается только ёмкость cortex.

## Модели

- 3M: 2,998,620 params; d=192; 6 layers; 6 heads; FF=570.
- 8M: 7,966,688 params; d=304; 8 layers; 8 heads; FF=768.
- 11M: 11,576,208 params; d=336; 10 layers; 8 heads; FF=840.

## Held-out BPB, context=48

### SynTagRus

- 3M: `2.191360`
- 8M: `2.076233` — **−5.254% vs 3M**
- 11M: `1.983113` — **−9.503% vs 3M**, **−4.485% vs 8M**

### GSD

- 3M: `2.561211`
- 8M: `2.400362` — **−6.280% vs 3M**
- 11M: `2.280639` — **−10.955% vs 3M**, **−4.988% vs 8M**

Кривая при одинаковом очень малом data exposure строго монотонна: `11M < 8M < 3M` по BPB.

## Training tail

Последние 128 step mean token-NLL:

- 3M: `7.31267`
- 8M: `6.97633`
- 11M: `6.86973`

Wall time:

- 3M: 168.6 s
- 8M: 343.7 s
- 11M: 434.0 s

## Raw generation

Все три модели после всего 393k target tokens всё ещё недообучены для свободной речи.

- 3M greedy быстро схлопывается в `что, что, что...`.
- 8M и 11M показывают более разнообразные локальные траектории, но также образуют повторные attractors (`ха`, `еее`, `что-то`, `конкон`).
- nucleus sampling повышает формальную lexical diversity, но осмысленная связная русская речь ещё не сформирована.

Следовательно, pilot подтверждает **capacity effect**, но не доказывает готовность 11M как языковой модели.

## Решение

1. Основной trusted R5.24 8192-step matched sweep должен подтвердить/опровергнуть сохранение монотонной кривой при 8× большем data exposure.
2. Параллельно R5.26 продолжает 11.6M pilot на document-disjoint diverse Russian corpus без graph-prefix и без left-UNK.
3. Exact logic/cyber остаётся отдельным Authority organ; не смешивается с ordinary surface LM.
