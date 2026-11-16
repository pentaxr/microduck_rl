import sys, json
sys.path.insert(0, sys.argv[1])
import numpy as np
import robot as r
out = {}
for k, j in list(r.J.items()):
    out[k] = dict(name=j.name, o=j.o.tolist(), R=j.R.tolist(), rng=list(j.rng), sleeve=j.sleeve, yoke=j.yoke)
for k in r.SIDE_JOINTS:
    jr = r.mirror_joint(r.J[k])
    out[k + '_R'] = dict(name=jr.name, o=jr.o.tolist(), R=jr.R.tolist(), rng=list(jr.rng), sleeve=jr.sleeve, yoke=jr.yoke)
out['_BAYS'] = {k: [list(map(float, a)), list(map(float, b))] for k, (a, b) in r.BAYS.items()}
json.dump(out, open(sys.argv[2], 'w'), indent=1)
for k, v in out.items(): print(k, v if k.startswith('_') else (v['name'], v['o'], v['sleeve'], v['yoke'], v['rng'], np.array(v['R'])[:,2].round(3)))
