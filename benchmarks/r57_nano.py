#!/usr/bin/env python3
import r57_concept_graph_language as r57

r57.SCREEN_STEPS = 16
r57.SCREEN_BATCH = 8
r57.LONG_STEPS = 64
r57.LONG_BATCH = 8
r57.MODES = ('A_PLAIN', 'D_LOGIC_CYBER')
r57.PROMPTS = [
    'Наука развивается потому, что',
    'Если система получила сигнал, то',
]

_original_aux = r57.aux_generation_accuracy
def _fast_aux(model, mode, tok, memory, kind, seed, n=32):
    return _original_aux(model, mode, tok, memory, kind, seed, n=4)
r57.aux_generation_accuracy = _fast_aux

if __name__ == '__main__':
    r57.main()
