from Box2D.b2 import circleShape, polygonShape
import json

class B2DJsonLoader(object):
    def __init__(self, file_path):
        self.rigid_bodies = []
        with open(file_path) as json_file:
            str_data = json.load(json_file)
            self._read_json(str_data)

    def attach_fixtures(self, name, body, fixture_def, scale):
        rigid_body_to_attach = next(rb for rb in self.rigid_bodies if rb.name == name)
        if rigid_body_to_attach is None:
            raise Exception("No rigidBody with this name was found.")

        _origin = rigid_body_to_attach.origin*scale

        for polygon in rigid_body_to_attach.polygons:
            _vertices = []
            for vertex in polygon.vertices:
                _vertex = vertex*scale
                _vertices.append((_vertex[0] - _origin[0], _vertex[1] - _origin[1]))
            fixture_def.shape = polygonShape(vertices=_vertices)
            body.CreateFixture(fixture_def)

        for circle in rigid_body_to_attach.circles:
            _center = circle.center*scale
            center = (_center[0] - _origin[0], _center[1] - _origin[1])
            radius = circle.radius*scale
            fixture_def.shape = circleShape(center, radius)
            body.CreateFixture(fixture_def)


    def _read_json(self, data):
        for rigid_body in data["rigidBodies"]:
            loaded_rigid_body = self._load_rigid_body(rigid_body)
            self.rigid_bodies.append(loaded_rigid_body)

    def _load_rigid_body(self, body_infos):
        rigid_body = self.RigidBody(body_infos["name"],
                                    body_infos["imagePath"],
                                    (body_infos["origin"]["x"], body_infos["origin"]["y"]))

        for polygon in body_infos["polygons"]:
            current_polygon = self.Polygon()
            for vertices in polygon:
                current_polygon.vertices.append((vertices["x"], vertices["y"]))
            rigid_body.polygons.append(current_polygon)

        for circle in body_infos["circles"]:
            current_circle = self.Cicle((circle["cx"], circle["cy"]),
                                        circle["radius"])
            rigid_body.circles.append(current_circle)

        return rigid_body


    class RigidBody(object):
        def __init__(self, name, image_path, origin):
            self.name = name
            self.image_path = image_path
            self.origin = origin
            self.polygons = []
            self.circles = []

    class Polygon(object):
        def __init__(self):
            self.vertices = []

    class Cicle(object):
        def __init__(self, center, radius):
            self.center = center
            self.radius = radius