from topax import show_part
from topax.sdfs import box, sphere, union

show_part(
    union(
        sphere(0.5, x=-0.5),
        sphere(0.5, x=0.5),
        box([0.5, 0.5, 0.5])
    ),
    [0.4, 0.6, 0.4]
)

show_part(
    sphere(0.5, z=0.5),
    [0.4, 0.6, 0.8]
)
