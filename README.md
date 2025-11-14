# Topax - A Full Stack Implicit Geometry CAD Package for Python

[![License](https://img.shields.io/github/license/andycate/topax)](LICENSE)

Topax is a CAD package specifically designed for working with implicit
geometry and programmatic CAD generation. It pulls a lot of ideas from
previous work such as [libfive](https://github.com/libfive/libfive). 

What makes this tool special is that it comes with a live viewer/editor
that updates in real-time as python code is changed. Objects are rendered
using direct ray-marching and JIT compiled OpenGL shaders.

**This tool is under active development!** Please consider contributing if
you find this interesting and would like to help out.

## Quick start

For now, this package needs to be installed by cloning the repo. You can
clone the repo with `git clone https://github.com/andycate/topax.git` and
then install the package by running `pip install -e .` inside the cloned
directory. Make sure to use a python virtual environment!

NOTE: for the stellarator examples, you will need to have the package `desc-opt` installed.

You can check out the examples in the `examples/` folder by running 
`topax --auto_reload examples/<file>.py`. Now, try changing some of the parameters
in that file. When you save the file, you'll notice that the model is 
instantly updated.

You can add parameters which will appear as sliders which can be adjusted in 
real time on the second window. Check out the examples to see how it works.

### Keyboard Shortcuts
- `m`: change lighting mode (cycles between three)
- `f`: front view
- `t`: top view
- `r`: right view

## The vision + Resources
Not really sure where I'm going with this project, but I found it interesting
to build! I think it would be cool to do a few different things:

1) Optimize the shaders that are generated to increase rendering performance
2) Add a way to make assemblies of single parts
3) Add lots more standard SDFs to the library
4) Add a way to specify variable parameters, and have those parameters adjustable as sliders or buttons on the CAD viewer window
5) Add some kind of interactive manipulation of objects in the CAD viewer window
6) Maybe add a way to run simulations and then use generated fields to manipulate geometry?
7) Some kind of geometry optimization???

Inigo Quilez has written lots of great articles about implicit modeling on [his website](https://iquilezles.org/). So has [Matt Keeter](https://www.mattkeeter.com/), who also developed libfive. Also, there is a commercial CAD package called [nTop](https://www.ntop.com/), which I believe is built on top of libfive.

## Disclaimer
I know that there are certain tools that already exist which do these things, and I'm not trying to replace them, I'm mostly doing this as a learning experience! And maybe some people will find it useful along the way.
