from topax.sdfs import box, sphere, union

def make_part():
    return union(
        sphere(0.5, x=-0.5),
        sphere(0.5, x=0.5),
        box([0.5, 0.5, 0.5])
    )
