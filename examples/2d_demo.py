from topax import show_part
from topax.sdfs import circle, union, offset, rectangle, tlp, slice_2d, gyroid, rotate, extrude

radius = 10.0
slice_angle = 17.0
outer_shell = 0.5

show_part(
    # circle(0.5).sub(circle(0.5).t(x=0.3)).o(-0.05),
    # tlp(tlp(rectangle(0.1, 0.2).o(5e-2), 4e-1, 1), 5e-1, 1, 'y'),
    extrude(circle(radius).i(
        slice_2d(
            gyroid().r(slice_angle, axis='x')
        )
    ).o(0.1).u(
        circle(radius+outer_shell).sub(circle(radius))
    ), 5.0),
    [0.4, 0.6, 0.8]
)
