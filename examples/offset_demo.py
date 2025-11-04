from topax import show_part
from topax.sdfs_old import box, sphere, cylinder, gyroid, offset, intersect, tlp, cp

show_part(
    #tlp(offset(box(0.5), 0.08), [1.5, 2.5, 2.5], [2, 2, 1]),
    tlp(
        cp(
            offset(
                intersect(box(0.5), offset(gyroid(scale=8., fill=0.08, thickness=0.33), -0.16)), 
                amount=0.2
            ), 
            r=2.5,
            nrep=3
        ),
        spacing=[0, 0, 2],
        nrep=[0, 0, 1]
    ),
    [0.3, 0.5, 0.1]
)
