from teachDRL.gym_flowers.envs.bodies.AbstractBody import AbstractBody
from teachDRL.gym_flowers.envs.bodies.BodyTypesEnum import BodyTypesEnum

class SwimmerAbstractBody(AbstractBody):
    def __init__(self, scale, motors_torque, density, nb_steps_outside_water):
        super(SwimmerAbstractBody, self).__init__(scale, motors_torque)

        self.body_type = BodyTypesEnum.SWIMMER
        self.nb_steps_can_survive_outside_water = nb_steps_outside_water
        self.DENSITY = density - 0.01