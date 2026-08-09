#!/usr/bin/env python3
import r57_concept_graph_language as r57

# Same architecture/data/evaluation path; only training budget and screening breadth are reduced.
r57.SCREEN_STEPS = 128
r57.SCREEN_BATCH = 8
r57.LONG_STEPS = 768
r57.LONG_BATCH = 8
r57.MODES = ('A_PLAIN', 'D_LOGIC_CYBER')

if __name__ == '__main__':
    r57.main()
