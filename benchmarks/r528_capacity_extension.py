#!/usr/bin/env python3
import r524_capacity_sweep as cap
cap.CONFIGS.update({
    '20M': dict(d=448,h=8,l=10,f=1120,expected=19950784),
    '30M': dict(d=512,h=8,l=12,f=1280,expected=30480384),
})
if __name__=='__main__':
    cap.main()
