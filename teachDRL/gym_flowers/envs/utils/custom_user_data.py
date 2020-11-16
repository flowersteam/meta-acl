from enum import Enum

class CustomUserDataObjectTypes(Enum):
    BODY_OBJECT = 0
    WATER = 1
    TERRAIN = 2
    GRIP_TERRAIN = 3
    MOTOR = 4
    BODY_SENSOR = 5
    SENSOR_GRIP_TERRAIN = 6


class CustomUserData(object):
    def __init__(self, name, object_type):
        self.name = name
        self.object_type = object_type

class CustomMotorUserData(CustomUserData):
    def __init__(self, speed_control, check_contact, angle_correction = 0.0, contact_body = None):
        super(CustomMotorUserData, self).__init__("motor", CustomUserDataObjectTypes.MOTOR)
        self.speed_control = speed_control
        self.check_contact = check_contact
        self.angle_correction = angle_correction
        self.contact_body = contact_body

class CustomBodyUserData(CustomUserData):
    def __init__(self, check_contact, is_contact_critical = False,
                 name = "body_part", object_type = CustomUserDataObjectTypes.BODY_OBJECT):
        super(CustomBodyUserData, self).__init__(name, object_type)
        self.check_contact = check_contact
        self.is_contact_critical = is_contact_critical
        self.has_contact = False


class CustomBodySensoUserData(CustomBodyUserData):
    def __init__(self, check_contact, is_contact_critical = False,
                 name = "body_part",):
        super(CustomBodySensoUserData, self).__init__(check_contact=check_contact,
                                                      is_contact_critical=is_contact_critical,
                                                      name=name,
                                                      object_type=CustomUserDataObjectTypes.BODY_SENSOR)
        self.has_joint = False
        self.ready_to_attach = False