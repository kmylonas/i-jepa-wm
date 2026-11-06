"""Fixed hand-positioned radial training and generalization coordinates."""

from collections import defaultdict

import numpy as np

from src.mylonas_ijepa_wm.curriculum_local_geometry import profiles


def make_design(bounds=(4.001,59.999)):
    goal=np.array([50.,42.])
    points,lookup=[],{}
    def valid(p):
        p=np.asarray(p)
        return bool((p>=bounds[0]-1e-6).all() and (p<=bounds[1]+1e-6).all() and p[0]>35)
    def intern(p):
        key=tuple(map(float,np.asarray(p,dtype=np.float32)))
        if key not in lookup:
            lookup[key]=len(points);points.append(list(key))
        return lookup[key]
    def position(radius,angle):
        theta=np.deg2rad(angle)
        return goal+radius*np.array([np.cos(theta),np.sin(theta)])
    goal_id=intern(goal)
    training=[]
    for radius in range(2,35,4):
        for angle in range(0,360,15):
            p=position(radius,angle)
            if valid(p):training.append(dict(radius=radius,angle=angle,point_id=intern(p)))
    stages=[dict(index=i,outer_radius=r,training_ids=[t['point_id'] for t in training if t['radius']<=r])
            for i,r in enumerate((6,14,22,34),1)]
    curves=[]
    for kind,angles in (('trained_direction',np.arange(0,360,15)),
                        ('withheld_direction',np.arange(7.5,360,15))):
        for angle in angles:
            radii=[r for r in range(34,1,-2) if valid(position(r,angle))]
            curves.append(dict(kind=kind,angle=float(angle),radii=radii,
                point_ids=[intern(position(r,angle)) for r in radii]))
    groups=defaultdict(list)
    for angle in range(0,360,15):
        ray=sorted([t for t in training if t['angle']==angle],key=lambda t:t['radius'])
        for inner,outer in zip(ray,ray[1:]):
            generated=profiles(points[outer['point_id']],points[inner['point_id']],goal,bounds)
            family=dict(angle=angle,inner_radius=inner['radius'],outer_radius=outer['radius'],
                rejected=generated['rejected'],profiles=[dict(offset=p['offset'],
                    point_ids=[intern(xy) for xy in p['points']]) for p in generated['valid']])
            groups[outer['radius']].append(family)
    # Selection depends only on geometry; round-robin radius intervals prevents
    # the dense near-goal rays from monopolizing the example cohort.
    rng=np.random.default_rng(20261106)
    for rows in groups.values():rng.shuffle(rows)
    families=[]
    while len(families)<50 and any(groups.values()):
        for radius in sorted(groups):
            if groups[radius] and len(families)<50:
                family=groups[radius].pop();family['family_id']=len(families);families.append(family)
    # Geometry construction may intern unselected families. Keep only the
    # actual fixed experiment inputs before any encoding occurs.
    xs=[35.001]+list(range(36,60,2))+[bounds[1]]
    ys=[bounds[0]]+list(range(6,60,2))+[bounds[1]]
    grid_ids=[intern([x,y]) for y in ys for x in xs]
    used={goal_id,*grid_ids,*[t['point_id'] for t in training]}
    for row in curves:used.update(row['point_ids'])
    for family in families:
        for profile in family['profiles']:used.update(profile['point_ids'])
    old_ids=sorted(used);remap={old:new for new,old in enumerate(old_ids)}
    for row in training:row['point_id']=remap[row['point_id']]
    for row in stages:row['training_ids']=[remap[i] for i in row['training_ids']]
    for row in curves:row['point_ids']=[remap[i] for i in row['point_ids']]
    for family in families:
        for row in family['profiles']:row['point_ids']=[remap[i] for i in row['point_ids']]
    return dict(schema_version=1,goal=goal.tolist(),goal_id=remap[goal_id],bounds=list(bounds),
        points=[points[i] for i in old_ids],training=training,stages=stages,curves=curves,families=families,
        grid=dict(xs=xs,ys=ys,point_ids=[remap[i] for i in grid_ids]),
        matched_family_ids=[f['family_id'] for f in families if len(f['profiles'])==7])
