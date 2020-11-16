from gym.envs.classic_control import rendering
try:
    import pyglet
except ImportError as e:
    raise ImportError('''
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    ''')

try:
    from pyglet.gl import *
except ImportError as e:
    raise ImportError('''
    Error occurred while running `from pyglet.gl import *`
    HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'.
    If you're running on a server, you may need a virtual frame buffer; something like this should work:
    'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'
    ''')

import numpy as np
class CustomViewer(rendering.Viewer):
    def __init__(self, width, height, display=None):
        super(CustomViewer, self).__init__(width, height, display)
        self.window.set_visible(False)

    def get_rgb_array(self):
        return super().render(return_rgb_array=True)

    def render(self, return_rgb_array=False):
        if not self.window.visible:
            self.window.set_visible(True)
        return super().render(return_rgb_array)

class EmptyViewer(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.isopen = True
        self.geoms = []
        self.onetime_geoms = []
        self.transform = rendering.Transform()

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width/(right-left)
        scaley = self.height/(top-bottom)
        self.transform = rendering.Transform(
            translation=(-left*scalex, -bottom*scaley),
            scale=(scalex, scaley))

    def add_geom(self, geom):
        self.geoms.append(geom)

    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)

    def render(self):
        glClearColor(1,1,1,1)
        self.transform.enable()
        for geom in self.geoms:
            geom.render()
        for geom in self.onetime_geoms:
            geom.render()
        self.transform.disable()

        buffer = pyglet.image.get_buffer_manager().get_color_buffer()
        image_data = buffer.get_image_data()
        arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
        # In https://github.com/openai/gym-http-api/issues/2, we
        # discovered that someone using Xmonad on Arch was having
        # a window of size 598 x 398, though a 600 x 400 window
        # was requested. (Guess Xmonad was preserving a pixel for
        # the boundary.) So we use the buffer height/width rather
        # than the requested one.
        arr = arr.reshape(buffer.height, buffer.width, 4)
        arr = arr[::-1,:,0:3]
        self.onetime_geoms = []
        return arr

    # Convenience
    def draw_circle(self, radius=10, res=30, filled=True, **attrs):
        geom = rendering.make_circle(radius=radius, res=res, filled=filled)
        rendering._add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polygon(self, v, filled=True, **attrs):
        geom = rendering.make_polygon(v=v, filled=filled)
        rendering._add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polyline(self, v, **attrs):
        geom = rendering.make_polyline(v=v)
        rendering._add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_line(self, start, end, **attrs):
        geom = rendering.Line(start, end)
        rendering._add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom