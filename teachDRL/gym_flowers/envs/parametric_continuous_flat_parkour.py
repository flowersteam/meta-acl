# Parametric Walker continuous environment
#
# Reward is given for moving forward, total 300+ points up to the far end. If the robot falls,
# it gets -100. Applying motor torque costs a small amount of points, more optimal agent
# will get better score.
#
# State consists of hull angle speed, angular velocity, horizontal speed, vertical speed,
# position of joints and joints angular speed, legs contact with ground, and 10 lidar
# rangefinder measurements. There's no coordinates
# in the state vector.
#
# Initially Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

#region Imports

import math

import Box2D
import gym
import numpy as np
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape)
from gym import spaces
from gym.utils import seeding, EzPickle
from teachDRL.gym_flowers.envs.Box2D_dynamics.water_dynamics import WaterDynamics, WaterContactDetector
from teachDRL.gym_flowers.envs.Box2D_dynamics.climbing_dynamics import ClimbingDynamics, ClimbingContactDetector
from teachDRL.gym_flowers.envs.bodies.BodiesEnum import BodiesEnum
from teachDRL.gym_flowers.envs.bodies.BodyTypesEnum import BodyTypesEnum
from teachDRL.gym_flowers.envs.utils.custom_user_data import CustomUserDataObjectTypes, CustomUserData


#endregion

#region Utils
class ContactDetector(WaterContactDetector, ClimbingContactDetector):
    def __init__(self, env):
        super(ContactDetector, self).__init__()
        self.env = env
        self.joints_to_add = []
    def BeginContact(self, contact):
        bodies = [contact.fixtureA.body, contact.fixtureB.body]
        if any([body.userData.object_type == CustomUserDataObjectTypes.WATER for body in bodies]):
            WaterContactDetector.BeginContact(self, contact)
        elif any([body.userData.object_type == CustomUserDataObjectTypes.BODY_SENSOR for body in bodies]):
            ClimbingContactDetector.BeginContact(self, contact)
        else:
            if contact.fixtureA.sensor or contact.fixtureB.sensor:
                return
            for idx, body in enumerate(bodies):
                if body.userData.object_type == CustomUserDataObjectTypes.BODY_OBJECT and body.userData.check_contact:
                    body.userData.has_contact = True
                    other_body = bodies[(idx + 1) % 2]
                    # Authorize climbing bodies to touch climbing parts
                    if body.userData.is_contact_critical and \
                            not (other_body.userData.object_type == CustomUserDataObjectTypes.GRIP_TERRAIN and
                                         self.env.agent_body.body_type == BodyTypesEnum.CLIMBER):
                        self.env.critical_contact = True

    def EndContact(self, contact):
        bodies = [contact.fixtureA.body, contact.fixtureB.body]
        if any([body.userData.object_type == CustomUserDataObjectTypes.WATER for body in bodies]):
            WaterContactDetector.EndContact(self, contact)
        elif any([body.userData.object_type == CustomUserDataObjectTypes.BODY_SENSOR for body in bodies]):
            ClimbingContactDetector.EndContact(self, contact)
        else:
            for body in [contact.fixtureA.body, contact.fixtureB.body]:
                if body.userData.object_type == CustomUserDataObjectTypes.BODY_OBJECT and body.userData.check_contact:
                    body.userData.has_contact = False

    def Reset(self):
        WaterContactDetector.Reset(self)
        ClimbingContactDetector.Reset(self)


class LidarCallback(Box2D.b2.rayCastCallback):
    def __init__(self, agent_mask_filter):
        Box2D.b2.rayCastCallback.__init__(self)
        self.agent_mask_filter = agent_mask_filter
        self.fixture = None
        self.is_water_detected = False
    def ReportFixture(self, fixture, point, normal, fraction):
        if (fixture.filterData.categoryBits & self.agent_mask_filter) == 0 or fixture.body.userData.object_type == CustomUserDataObjectTypes.SENSOR_GRIP_TERRAIN:
            return -1
        self.p2 = point
        self.fraction = fraction
        self.is_water_detected = True if fixture.body.userData.object_type == CustomUserDataObjectTypes.WATER else False
        return fraction

#endregion

#region Constants

FPS    = 50
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well
VIEWPORT_W = 600
VIEWPORT_H = 430

NB_LIDAR = 10
LIDAR_RANGE   = 160/SCALE

INITIAL_RANDOM = 5

TERRAIN_STEP   = 14/SCALE
TERRAIN_LENGTH = 200     # in steps
TERRAIN_HEIGHT = VIEWPORT_H/SCALE/4
TERRAIN_GRASS    = 10    # low long are grass spots, in steps
INITIAL_TERRAIN_STARTPAD = 20 # in steps
FRICTION = 2.5

WATER_DENSITY = 4.0
NB_FIRST_STEPS_HANG = 5
ROOF_HEIGHT = 10

#endregion

class ParametricContinuousWalker(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    def __init__(self, water_level, agent_body_type, **walker_args):
        super(ParametricContinuousWalker, self).__init__()

        # Seed env and init Box2D
        self.seed()
        self.viewer = None

        self.contact_listener = ContactDetector(self)
        self.world = Box2D.b2World(contactListener=self.contact_listener)
        #self.world = Box2D.b2World()
        self.terrain = []
        self.water_dynamics = WaterDynamics(self.world.gravity)
        self.water_level = water_level
        self.climbing_dynamics = ClimbingDynamics()

        self.prev_shaping = None

        # Create walker
        body_type = BodiesEnum.get_body_type(agent_body_type)
        if body_type == BodyTypesEnum.SWIMMER or body_type == BodyTypesEnum.AMPHIBIAN:
            self.agent_body = BodiesEnum[agent_body_type].value(SCALE, density=WATER_DENSITY, **walker_args)
        else:
            self.agent_body = BodiesEnum[agent_body_type].value(SCALE, **walker_args)

        #TODO : Check impact on REWARD !!!
        self.TERRAIN_STARTPAD = INITIAL_TERRAIN_STARTPAD if \
            self.agent_body.AGENT_WIDTH / TERRAIN_STEP + 5 <= INITIAL_TERRAIN_STARTPAD else \
            self.agent_body.AGENT_WIDTH / TERRAIN_STEP + 5  # in steps
        self.create_terrain_fixtures()

        # Set info / action spaces
        self._generate_agent()  # To get state / action sizez
        agent_action_size = self.agent_body.get_action_size()
        self.action_space = spaces.Box(np.array([-1] * agent_action_size),
                                       np.array([1] * agent_action_size), dtype=np.float32)

        agent_state_size = self.agent_body.get_state_size()
        high = np.array([np.inf] * (agent_state_size +
                                    5 +  # head infos (including water)
                                    NB_LIDAR + 1))  # lidars infos + % of water detection
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # Gather parameters for procedural track generation, make sure to call this before each new episode
    def set_environment(self, climbing_surface_size=0.5, gap_pos=None, gap_size=5.0,
                        obstacle_spacing=None):

        # ADDING STUMPS
        stump_height = None
        #self.obstacle_spacing = max(0.01, obstacle_spacing) if obstacle_spacing is not None else 8.0
        #self.stump_height = [stump_height, 0.1] if stump_height is not None else None

        self.climbing_surface_size = climbing_surface_size
        # UPDATE: now there is creepers and climbing surface
        # if climbing_surface_size is not None:
        #     self.creepers = False
        # else:
        self.creepers = True
        self.creepers_width = TERRAIN_STEP*0.8
        #self.creepers_height = [creepers_height, 0.2]
        # compute stump and creeper heights from w.r.t gap
        #print('gap_pos is {}, obstacle_spacing is {}'.format(gap_pos, obstacle_spacing))
        assert(10 >= gap_pos >= 2.5)
        self.creepers_height = [max(0,ROOF_HEIGHT - gap_pos - (gap_size/2)), 0.2]
        self.stump_height = [max(gap_pos - (gap_size/2),0), 0.05]

        self.obstacle_spacing = max(0.01, obstacle_spacing)

    def _destroy(self):
        # if not self.terrain: return
        self.world.contactListener = None
        for t in self.terrain:
            self.world.DestroyBody(t)
        self.terrain = []

        self.agent_body.destroy(self.world)

    def reset(self):
        self.world.contactListener = None
        self.contact_listener.Reset()
        self._destroy()
        self.world.contactListener = self.contact_listener
        self.critical_contact = False
        self.prev_shaping = None
        self.scroll = 0.0
        self.lidar_render = 0

        self.generate_game()

        self.drawlist = self.terrain + self.agent_body.get_elements_to_render()

        self.lidar = [LidarCallback(self.agent_body.reference_head_object.fixtures[0].filterData.maskBits)
                      for _ in range(NB_LIDAR)]

        actions_to_play = np.array([0] * self.action_space.shape[0])
        if self.agent_body.body_type == BodyTypesEnum.CLIMBER:
            # Init climber
            y_diff = 0
            for i in range(len(self.agent_body.sensors)):
                actions_to_play[len(actions_to_play) - i - 1] = 1
                # Hang sensor
                sensor = self.agent_body.sensors[len(self.agent_body.sensors) - i - 1]
                if y_diff == 0:
                    y_diff = TERRAIN_HEIGHT + (ROOF_HEIGHT - 0.01) - sensor.position[1]
                sensor.position = (sensor.position[0],
                                   TERRAIN_HEIGHT + (ROOF_HEIGHT - 0.01))

            for body_part in self.agent_body.body_parts:
                body_part.position = (body_part.position[0],
                                      body_part.position[1] + y_diff)

            for i in range(NB_FIRST_STEPS_HANG):
                self.step(actions_to_play)

        return self.step(actions_to_play)[0]

    def step(self, action):
        self.agent_body.activate_motors(action)

        if self.agent_body.body_type == BodyTypesEnum.CLIMBER:
            self.climbing_dynamics.before_step_climbing_dynamics(action, self.agent_body, self.world)

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        if self.agent_body.body_type == BodyTypesEnum.CLIMBER:
            self.climbing_dynamics.after_step_climbing_dynamics(self.world.contactListener, self.world)

        self.water_dynamics.calculate_forces(self.world.contactListener.fixture_pairs)

        head = self.agent_body.reference_head_object
        pos = head.position
        vel = head.linearVelocity

        lidar_angular_spread = 1.5
        lidar_offset = 0
        if self.agent_body.body_type == BodyTypesEnum.CLIMBER:
            lidar_angular_spread = 2.3
            lidar_offset = 1.5

        for i in range(NB_LIDAR):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin((lidar_angular_spread * i / NB_LIDAR)+lidar_offset) * LIDAR_RANGE,
                pos[1] - math.cos((lidar_angular_spread * i / NB_LIDAR)+lidar_offset) * LIDAR_RANGE)
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)
        state = [
            head.angle,  # Normal angles up to 0.5 here, but sure more is possible.
            2.0 * head.angularVelocity / FPS,
            0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,  # Normalized to get -1..1 range
            0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,
            pos.y <= self.water_level
        ]

        # add leg-related state
        state.extend(self.agent_body.get_motors_state())

        if self.agent_body.body_type == BodyTypesEnum.CLIMBER:
            state.extend(self.agent_body.get_sensors_state())

        nb_of_water_detected = 0
        for lidar in self.lidar:
            state.append(lidar.fraction)
            if lidar.is_water_detected:
                nb_of_water_detected += 1
        state.append(nb_of_water_detected / NB_LIDAR)  # percentage of lidars that detect water

        self.scroll = pos[0] - VIEWPORT_W / SCALE / 5  # 1 = grass

        shaping = 130 * pos[
            0] / SCALE  # moving forward is a way to receive reward (normalized to get 300 on completion)
        if not (
                    hasattr(self.agent_body,
                            "remove_reward_on_head_angle") and self.agent_body.remove_reward_on_head_angle):
            shaping -= 5.0 * abs(
                state[0])  # keep head straight, other than that and falling, any behavior is unpunished

        reward = 0
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        for a in action:
            # reward -= self.walker_body.TORQUE_PENALTY * self.walker_body.MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)
            reward -= self.agent_body.TORQUE_PENALTY * 80 * np.clip(np.abs(a), 0, 1)  # 80 => Original torque
            # normalized to about -50.0 using heuristic, more optimal agent should spend less

        done = False
        if self.critical_contact or pos[0] < 0:
            reward = -100
            done = True
        if pos[0] > (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP:
            done = True

        return np.array(state), reward, done, {}

    def render(self, mode='human', draw_lidars=True):
        #self.scroll = 1
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
        self.viewer.set_bounds(self.scroll, VIEWPORT_W/SCALE + self.scroll, 0, VIEWPORT_H/SCALE)

        self.viewer.draw_polygon( [
            (self.scroll,                  0),
            (self.scroll+VIEWPORT_W/SCALE, 0),
            (self.scroll+VIEWPORT_W/SCALE, VIEWPORT_H/SCALE),
            (self.scroll,                  VIEWPORT_H/SCALE),
            ], color=(0.9, 0.9, 1.0) )
        for poly,x1,x2 in self.cloud_poly:
            if x2 < self.scroll/2: continue
            if x1 > self.scroll/2 + VIEWPORT_W/SCALE: continue
            self.viewer.draw_polygon( [(p[0]+self.scroll/2, p[1]) for p in poly], color=(1,1,1))
        for poly, color in self.terrain_poly:
            if poly[1][0] < self.scroll: continue
            if poly[0][0] > self.scroll + VIEWPORT_W/SCALE: continue
            self.viewer.draw_polygon(poly, color=color)

        for obj in self.drawlist:
            color1 = obj.color1
            color2 = obj.color2
            if obj.userData.object_type == CustomUserDataObjectTypes.BODY_SENSOR and obj.userData.has_joint: # Color sensors when attached
                color1 = (1.0, 1.0, 0.0)
                color2 = (1.0, 1.0, 0.0)
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 30, color=color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 30, color=color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=color2, linewidth=2)

        # Draw lidars
        if draw_lidars:
            for i in range(len(self.lidar)):
                l = self.lidar[i]
                self.viewer.draw_polyline([l.p1, l.p2], color=(1, 0, 0), linewidth=1)

        flagy1 = TERRAIN_HEIGHT
        flagy2 = flagy1 + 50/SCALE
        x = TERRAIN_STEP*3
        self.viewer.draw_polyline( [(x, flagy1), (x, flagy2)], color=(0,0,0), linewidth=2 )
        f = [(x, flagy2), (x, flagy2-10/SCALE), (x+25/SCALE, flagy2-5/SCALE)]
        self.viewer.draw_polygon(f, color=(0.9,0.2,0) )
        self.viewer.draw_polyline(f + [f[0]], color=(0,0,0), linewidth=2 )

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        self.world.contactListener = None
        self.contact_listener.Reset()
        self._destroy()
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    #region Fixtures Initialization
    # ------------------------------------------ FIXTURES INITIALIZATION ------------------------------------------

    def create_terrain_fixtures(self):
        self.fd_polygon = fixtureDef(
            shape=polygonShape(vertices=
                               [(0, 0),
                                (1, 0),
                                (1, -1),
                                (0, -1)]),
            friction=FRICTION,
            categoryBits=0x1,
            maskBits=0xFFFF
        )

        self.fd_edge = fixtureDef(
            shape=edgeShape(vertices=
                            [(0, 0),
                             (1, 1)]),
            friction=FRICTION,
            categoryBits=0x1,
            maskBits=0xFFFF
        )

        # Init default hexagon fixture and shape, used only for Hexagon Tracks
        self.fd_default_hexagon = fixtureDef(
            shape=polygonShape(vertices=
                               [(0, 0),
                                (1, 0),
                                (1, -1),
                                (0, -1)]),
            friction=FRICTION,
            categoryBits=0x1,
            maskBits=0xFFFF
        )
        self.default_hexagon = [(-0.5, 0), (-0.5, 0.25), (-0.25, 0.5), (0.25, 0.5), (0.5, 0.25), (0.5, 0)]

        self.fd_water = fixtureDef(
            shape=polygonShape(vertices=
                               [(0, 0),
                                (1, 0),
                                (1, -1),
                                (0, -1)]),
            density=WATER_DENSITY,
            isSensor=True,
        )

        self.fd_creeper = fixtureDef(
            shape=polygonShape(vertices=
                               [(0, 0),
                                (1, 0),
                                (1, -1),
                                (0, -1)]),
            isSensor=True,
        )
    #endregion

    # region Game Generation
    # ------------------------------------------ GAME GENERATION ------------------------------------------

    def generate_game(self):
        self._generate_terrain()
        self._generate_clouds()
        self._generate_agent()

    def _generate_terrain(self):
        y = TERRAIN_HEIGHT
        self.terrain = []
        self.terrain_x = []
        self.terrain_y = []
        x = 0
        max_x = TERRAIN_LENGTH * TERRAIN_STEP
        if self.creepers:
            space_from_precedent_creeper = self.obstacle_spacing

        # Generation of terrain
        while x < max_x:
            self.terrain_x.append(x)
            self.terrain_y.append(y)
            x += TERRAIN_STEP

        # Draw terrain
        self.terrain_poly = []

        # temporary storage used to make sure all roof stumps are created after all roof creepers (rendering detail)
        tmp_roof_stumps = []
        tmp_roof_creepers = []

        assert len(self.terrain_x) == len(self.terrain_y)
        for i in range(len(self.terrain_x) - 1):
            # Ground
            poly = [
                (self.terrain_x[i], self.terrain_y[i]),
                (self.terrain_x[i + 1], self.terrain_y[i + 1])
            ]
            self.fd_edge.shape.vertices = poly
            t = self.world.CreateStaticBody(
                fixtures=self.fd_edge,
                userData=CustomUserData("grass", CustomUserDataObjectTypes.TERRAIN))
            color = (0.3, 1.0 if (i % 2) == 0 else 0.8, 0.3)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (0.4, 0.6, 0.3)
            poly += [(poly[1][0], 0), (poly[0][0], 0)]
            self.terrain_poly.append((poly, color))

            # Ceiling
            poly = [
                (self.terrain_x[i], self.terrain_y[i] + ROOF_HEIGHT),
                (self.terrain_x[i + 1], self.terrain_y[i + 1] + ROOF_HEIGHT)
            ]
            self.fd_edge.shape.vertices = poly
            t = self.world.CreateStaticBody(
                fixtures=self.fd_edge,
                userData=CustomUserData("rock", CustomUserDataObjectTypes.GRIP_TERRAIN))
            color = (0, 0.25, 0.25)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (0.5, 0.5, 0.5)
            poly += [(poly[1][0], 20), (poly[0][0], 20)]
            self.terrain_poly.append((poly, color))

            # Ground Stumps, placed at creepers width
            #print(self.TERRAIN_STARTPAD)
            if self.terrain_x[i] >= (self.TERRAIN_STARTPAD/2.5):
                if space_from_precedent_creeper >= self.obstacle_spacing:
                    stump_height = max(0.05, self.np_random.normal(self.stump_height[0], self.stump_height[1]))

                    # creeper_width = self.np_random.normal(self.creepers_width[0], self.creepers_width[1])
                    poly = [
                        (self.terrain_x[i], self.terrain_y[i]),
                        (self.terrain_x[i] + self.creepers_width, self.terrain_y[i]),
                        (self.terrain_x[i] + self.creepers_width, self.terrain_y[i] + stump_height),
                        (self.terrain_x[i], self.terrain_y[i] + stump_height),
                    ]
                    self.fd_polygon.shape.vertices = poly
                    t = self.world.CreateStaticBody(
                        fixtures=self.fd_polygon,
                        userData=CustomUserData("grass", CustomUserDataObjectTypes.TERRAIN))
                    t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                    self.terrain.append(t)
                #     space_from_precedent_creeper = 0
                # else:
                #     space_from_precedent_creeper += self.terrain_x[i] - self.terrain_x[i - 1]

            # # Creepers with stumps inside
            if self.terrain_x[i] >= (self.TERRAIN_STARTPAD / 2.5):
                if space_from_precedent_creeper >= self.obstacle_spacing:
                    creeper_height = max(0.05,self.np_random.normal(self.creepers_height[0], self.creepers_height[1]))
                    # creeper_width = self.np_random.normal(self.creepers_width[0], self.creepers_width[1])
                    poly = [
                        (self.terrain_x[i], self.terrain_y[i] + ROOF_HEIGHT),
                        (self.terrain_x[i] + self.creepers_width, self.terrain_y[i] + ROOF_HEIGHT),
                        (self.terrain_x[i] + self.creepers_width, self.terrain_y[i] + ROOF_HEIGHT - creeper_height),
                        (self.terrain_x[i], self.terrain_y[i] + ROOF_HEIGHT - creeper_height),
                    ]
                    big_poly = [
                        (self.terrain_x[i] - (self.creepers_width), self.terrain_y[i] + ROOF_HEIGHT),
                        (self.terrain_x[i] + (self.creepers_width*2), self.terrain_y[i] + ROOF_HEIGHT),
                        (self.terrain_x[i] + (self.creepers_width*2), self.terrain_y[i] + ROOF_HEIGHT - creeper_height - self.creepers_width),
                        (self.terrain_x[i] - (self.creepers_width), self.terrain_y[i] + ROOF_HEIGHT - creeper_height - self.creepers_width),
                    ]


                    # then draw the roof stump at given creeper dimensions
                    self.fd_polygon.shape.vertices = poly
                    t = self.world.CreateStaticBody(
                        fixtures=self.fd_polygon,
                        userData=CustomUserData("rock", CustomUserDataObjectTypes.GRIP_TERRAIN))
                    t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                    #self.terrain.append(t)
                    #print('drawing')
                    tmp_roof_stumps.append(t)

                    # first draw bigger gripping creeper
                    self.fd_creeper.shape.vertices = big_poly
                    t = self.world.CreateStaticBody(
                        fixtures=self.fd_creeper,
                        userData=CustomUserData("creeper", CustomUserDataObjectTypes.SENSOR_GRIP_TERRAIN))
                    c = (0.391, 0.613, 0.398)
                    t.color1 = c
                    t.color2 = c
                    #self.terrain.append(t)
                    tmp_roof_creepers.append(t)


                    space_from_precedent_creeper = self.terrain_x[i] - self.terrain_x[i - 1]
                else:
                    space_from_precedent_creeper += self.terrain_x[i] - self.terrain_x[i - 1]

        # now add tmp roof stump and creepers
        self.terrain.extend(tmp_roof_stumps)
        self.terrain.extend(tmp_roof_creepers)

        # Climbing surface
        #if not self.creepers:
        if self.climbing_surface_size > 0:
            climbing_surface_poly = [
                (self.terrain_x[0], self.terrain_y[0] + ROOF_HEIGHT),
                (self.terrain_x[0], self.terrain_y[0] + ROOF_HEIGHT - self.climbing_surface_size),
                (self.terrain_x[len(self.terrain_x) - 1], self.terrain_y[len(self.terrain_y) - 1] + ROOF_HEIGHT - self.climbing_surface_size),
                (self.terrain_x[len(self.terrain_x) - 1], self.terrain_y[len(self.terrain_y) - 1] + ROOF_HEIGHT)
            ]
            self.fd_creeper.shape.vertices = climbing_surface_poly
            t = self.world.CreateStaticBody(
                fixtures=self.fd_creeper,
                userData=CustomUserData("creeper", CustomUserDataObjectTypes.SENSOR_GRIP_TERRAIN))
            # c = (0.453, 0.824, 0.859)
            c = (0.391, 0.613, 0.398)
            t.color1 = c
            t.color2 = c
            self.terrain.append(t)

        # Water
        if self.water_level > 0:
            water_poly = [
                (self.terrain_x[0], self.terrain_y[0]),
                (self.terrain_x[0], self.terrain_y[0] + self.water_level),
                (self.terrain_x[len(self.terrain_x) - 1], self.terrain_y[len(self.terrain_y) - 1] + self.water_level),
                (self.terrain_x[len(self.terrain_x) - 1], self.terrain_y[len(self.terrain_y) - 1])
            ]
            self.fd_water.shape.vertices = water_poly
            t = self.world.CreateStaticBody(
                fixtures=self.fd_water,
                userData=CustomUserData("water", CustomUserDataObjectTypes.WATER))
            # c = (0.453, 0.824, 0.859)
            c = (0.465, 0.676, 0.898)
            t.color1 = c
            t.color2 = c
            self.terrain.append(t)

        self.terrain.reverse()

    def _generate_clouds(self):
        # Sorry for the clouds, couldn't resist
        self.cloud_poly   = []
        for i in range(TERRAIN_LENGTH//20):
            x = self.np_random.uniform(0, TERRAIN_LENGTH)*TERRAIN_STEP
            y = VIEWPORT_H/SCALE*3/4
            poly = [
                (x+15*TERRAIN_STEP*math.sin(3.14*2*a/5)+self.np_random.uniform(0,5*TERRAIN_STEP),
                 y+ 5*TERRAIN_STEP*math.cos(3.14*2*a/5)+self.np_random.uniform(0,5*TERRAIN_STEP) )
                for a in range(5) ]
            x1 = min( [p[0] for p in poly] )
            x2 = max( [p[0] for p in poly] )
            self.cloud_poly.append( (poly,x1,x2) )

    def _generate_agent(self):
        init_x = TERRAIN_STEP*self.TERRAIN_STARTPAD/2
        if hasattr(self.agent_body, "old_morphology") and self.agent_body.old_morphology:
            init_y = TERRAIN_HEIGHT + 2 * self.agent_body.LEG_H
        else:
            init_y = TERRAIN_HEIGHT + self.agent_body.AGENT_CENTER_HEIGHT

        self.agent_body.draw(
            self.world,
            init_x,
            init_y,
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)
        )

    def _SET_VIEWPORT(self, width, height):
        global VIEWPORT_W, VIEWPORT_H, SCALE, TERRAIN_HEIGHT
        VIEWPORT_W = width
        VIEWPORT_H = height
        TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4

    #endregion