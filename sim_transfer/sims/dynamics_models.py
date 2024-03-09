import os
from abc import ABC, abstractmethod
from typing import NamedTuple, Union, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax import random, vmap
from jaxtyping import PyTree


class PendulumParams(NamedTuple):
    m: jax.Array = jnp.array(1.0)
    l: jax.Array = jnp.array(1.0)
    g: jax.Array = jnp.array(9.81)
    nu: jax.Array = jnp.array(0.0)
    c_d: jax.Array = jnp.array(0.0)


class CarParams(NamedTuple):
    """
    d_f, d_r : Represent grip of the car. Range: [0.015, 0.025]
    b_f, b_r: Slope of the pacejka. Range: [2.0 - 4.0].

    delta_limit: [0.3 - 0.5] -> Limit of the steering angle.

    c_m_1: Motor parameter. Range [0.2, 0.5]
    c_m_1: Motor friction, Range [0.00, 0.007]
    c_f, c_r: [1.0 2.0] # motor parameters: source https://web.mit.edu/drela/Public/web/qprop/motor1_theory.pdf,
    https://ethz.ch/content/dam/ethz/special-interest/mavt/dynamic-systems-n-control/idsc-dam/Lectures/Embedded
    -Control-Systems/LectureNotes/6_Motor_Control.pdf # or look at:
    https://video.ethz.ch/lectures/d-mavt/2021/spring/151-0593-00L/00718f4f-116b-4645-91da-b9482164a3c7.html :
    lecture 2 part 2
    c_m_1: max current of motor: [0.2 - 0.5] c_m_2: motor resistance due to shaft: [0.01 - 0.15]
    """
    m: Union[jax.Array, float] = jnp.array(1.65)  # [0.04, 0.08]
    i_com: Union[jax.Array, float] = jnp.array(2.78e-05)  # [1e-6, 5e-6]
    l_f: Union[jax.Array, float] = jnp.array(0.13)  # [0.025, 0.05]
    l_r: Union[jax.Array, float] = jnp.array(0.17)  # [0.025, 0.05]
    g: Union[jax.Array, float] = jnp.array(9.81)

    d_f: Union[jax.Array, float] = jnp.array(0.02)  # [0.015, 0.025]
    c_f: Union[jax.Array, float] = jnp.array(1.2)  # [1.0, 2.0]
    b_f: Union[jax.Array, float] = jnp.array(2.58)  # [2.0, 4.0]

    d_r: Union[jax.Array, float] = jnp.array(0.017)  # [0.015, 0.025]
    c_r: Union[jax.Array, float] = jnp.array(1.27)  # [1.0, 2.0]
    b_r: Union[jax.Array, float] = jnp.array(3.39)  # [2.0, 4.0]

    c_m_1: Union[jax.Array, float] = jnp.array(10.431917)  # [0.2, 0.5]
    c_m_2: Union[jax.Array, float] = jnp.array(1.5003588)  # [0.00, 0.007]
    c_d: Union[jax.Array, float] = jnp.array(0.0)  # [0.01, 0.1]
    steering_limit: Union[jax.Array, float] = jnp.array(0.19989373)
    use_blend: Union[jax.Array, float] = jnp.array(0.0)  # 0.0 -> (only kinematics), 1.0 -> (kinematics + dynamics)

    # parameters used to compute the blend ratio characteristics
    blend_ratio_ub: Union[jax.Array, float] = jnp.array([0.5477225575])
    blend_ratio_lb: Union[jax.Array, float] = jnp.array([0.4472135955])
    angle_offset: Union[jax.Array, float] = jnp.array([0.02791893])


class SergioParams(NamedTuple):
    lam: jax.Array = jnp.array(0.8)
    contribution_rates: jax.Array = jnp.array(2.0)
    basal_rates: jax.Array = jnp.array(0.0)
    power: jax.Array = jnp.array(2.0)
    graph: jax.Array = jnp.array(1.0)


class GreenHouseParams(NamedTuple):
    beta: Union[jax.Array, float] = jnp.array(0.01)  # Heat absorption efficiency
    gamma: Union[jax.Array, float] = jnp.array(0.067)  # apparent psychometric constant
    epsilon: Union[jax.Array, float] = jnp.array(3.0)  # Cover heat resistance ratio
    zeta: Union[jax.Array, float] = jnp.array(2.7060 * (10 ** (-5)))  # ventilation rate parameter
    eta: Union[jax.Array, float] = jnp.array(0.7)  # radiation conversion factor
    theta: Union[jax.Array, float] = jnp.array(4.02 * (10 ** (-5)))  # ventilation rate parameter 1
    kappa: Union[jax.Array, float] = jnp.array(5.03 * (10 ** (-5)))  # ventilation rate parameter 2
    lam: Union[jax.Array, float] = jnp.array(0.46153)  # pressure constant
    mu: Union[jax.Array, float] = jnp.array(1.4667)  # Molar weight fraction C02 CH20
    nu_2: Union[jax.Array, float] = jnp.array(3.68 * (10 ** (-5)))  # ventilation rate parameter 1
    xi: Union[jax.Array, float] = jnp.array(6.3233 * (10 ** (-5)))  # ventilation rate parameter 2
    rho_w: Union[jax.Array, float] = jnp.array(998)  # Density of water
    rho_a: Union[jax.Array, float] = jnp.array(1.29)  # Density of air
    sigma: Union[jax.Array, float] = jnp.array(7.1708 * (10 ** (-5)))  # ventilation rate parameter 3
    tau: Union[jax.Array, float] = jnp.array(3.0)  # pipe heat transfer coefficient 1
    nu: Union[jax.Array, float] = jnp.array(0.74783)  # pipe heat transfer coefficient 2
    chi: Union[jax.Array, float] = jnp.array(0.0156)  # ventilation rate parameter 4
    psi: Union[jax.Array, float] = jnp.array(7.4 * (10 ** (-5)))
    omega: Union[jax.Array, float] = jnp.array(0.622)  # Humidity ratio
    a1: Union[jax.Array, float] = jnp.array(0.611)  # Saturated vapour pressure parameter 1
    a2: Union[jax.Array, float] = jnp.array(17.27)  # Saturated vapour pressure parameter 2
    a3: Union[jax.Array, float] = jnp.array(239.0)  # Saturated vapour pressure parameter 3
    ap: Union[jax.Array, float] = jnp.array(314.16)  # Heating pipe outer surface area
    b1: Union[jax.Array, float] = jnp.array(2.7)  # buffer_coefficient
    cg: Union[jax.Array, float] = jnp.array(32 * (10 ** 3))  # green_house_heat_capacity
    cp_w: Union[jax.Array, float] = jnp.array(4180.0)  # specific_heat_water
    cs: Union[jax.Array, float] = jnp.array(120 * (10 ** 3))  # green_house_soil_heat_capacity
    cp_a: Union[jax.Array, float] = 1010  # air_specific_heat_water
    d1: Union[jax.Array, float] = jnp.array(2.1332 * (10 ** (-7)))  # plant development rate 1
    d2: Union[jax.Array, float] = jnp.array(2.4664 * (10 ** (-1)))  # plant development rate 2
    d3: Union[jax.Array, float] = jnp.array(20)  # plant development rate 3
    d4: Union[jax.Array, float] = jnp.array(7.4966 * (10 ** (-11)))  # plant development rate 4
    f: Union[jax.Array, float] = jnp.array(1.2)  # fruit assimilate requirment
    f1: Union[jax.Array, float] = jnp.array(8.1019 * (10 ** (-7)))  # fruit growth rate
    f2: Union[jax.Array, float] = jnp.array(4.6296 * (10 ** (-6)))  # fruit growth rate
    g1: Union[jax.Array, float] = jnp.array(20.3 * (10 ** (-3)))  # Leaf conductance parameter 1
    g2: Union[jax.Array, float] = jnp.array(0.44)  # Leaf conductance parameter 2
    g3: Union[jax.Array, float] = jnp.array(2.5 * (10 ** (-3)))  # Leaf conductance parameter 3
    g4: Union[jax.Array, float] = jnp.array(3.1 * (10 ** (-4)))  # Leaf conductance parameter 4
    gb: Union[jax.Array, float] = jnp.array(10 ** (-2))  # Boundary layer conductance
    kd: Union[jax.Array, float] = jnp.array(2.0)  # Soil to soil heat transfer coefficient
    kr: Union[jax.Array, float] = jnp.array(7.9)  # Roof heat transfer coefficient
    ks: Union[jax.Array, float] = jnp.array(5.75)  # Soil to air heat transfer coefficient
    l1: Union[jax.Array, float] = jnp.array(2.501 * (10 ** 6))  # Vaporisation energy coefficient 1
    l2: Union[jax.Array, float] = jnp.array(2.381 * (10 ** 3))  # Vaporisation energy coefficient 2
    m1: Union[jax.Array, float] = jnp.array(1.0183 * (10 ** (-3)))  # mass transfer parameter
    m2: Union[jax.Array, float] = jnp.array(0.33)  # Mass transfer parameter 2
    Mco2: Union[jax.Array, float] = jnp.array(0.0044)  # Molar mass CO2
    MF: Union[jax.Array, float] = jnp.array(1.157 * (10 ** (-7)))  # Fruit maintenance respiration coefficient
    ML: Union[jax.Array, float] = jnp.array(2.894 * (10 ** (-7)))  # Vegetative maintenance respiration coefficient
    mp: Union[jax.Array, float] = jnp.array(4.57)  # Watt to micromol conversion constant
    p1: Union[jax.Array, float] = jnp.array(-2.17 * (10 ** (-4)))  # Net photosynthesis parameter 1 -> check these
    p2: Union[jax.Array, float] = jnp.array(3.31 * (10 ** (-3)))  # Net photosynthesis parameter 2 -> check these
    p3: Union[jax.Array, float] = jnp.array(577.0)  # Net photosynthesis parameter 3
    p4: Union[jax.Array, float] = jnp.array(221.0)  # Net photosynthesis parameter 4
    p5: Union[jax.Array, float] = jnp.array(5 * (10 ** (-5)))  # Net photosynthesis parameter 5 -> check these
    patm: Union[jax.Array, float] = jnp.array(101.0)  # atmospheric pressure
    pm: Union[jax.Array, float] = jnp.array(2.2538 * (10 ** (-3)))  # Maximum photosynthesis rate
    qg: Union[jax.Array, float] = jnp.array(2.0)  # fruit growth rate parameter
    qr: Union[jax.Array, float] = jnp.array(2.0)  # maintenance respiration
    rg: Union[jax.Array, float] = jnp.array(8.3144)  # Gas constant
    s1: Union[jax.Array, float] = jnp.array(1.8407 ** (-4))  # saturated water vapour pressure curve slope parameter 1
    s2: Union[jax.Array, float] = jnp.array(
        9.7838 ** (10 ** (-4)))  # saturated water vapour pressure curve slope parameter 2
    s3: Union[jax.Array, float] = jnp.array(0.051492)  # saturated water vapour pressure curve slope parameter 3
    T0: Union[jax.Array, float] = jnp.array(273.15)  # conversion from Celsius to K
    Tg: Union[jax.Array, float] = jnp.array(20.0)  # growth rate temperature reference
    Td: Union[jax.Array, float] = jnp.array(10.0)  # Deep soil temperature
    Tr: Union[jax.Array, float] = jnp.array(25.0)  # Maintenance respiration reference temperature
    v: Union[jax.Array, float] = jnp.array(1.23)  # Vegetative assimilate requirement coefficient
    v1: Union[jax.Array, float] = jnp.array(1.3774)  # Vegetative fruit growth ratio parameter 1
    v2: Union[jax.Array, float] = jnp.array(-0.168)  # Vegetative fruit growth ratio parameter 2
    v3: Union[jax.Array, float] = jnp.array(19.0)  # Vegetative fruit growth ratio parameter 3
    vp: Union[jax.Array, float] = jnp.array(7.85)  # Heating pipe volume
    vg_ag: Union[jax.Array, float] = jnp.array(10.0)  # Average greenhouse height
    wr: Union[jax.Array, float] = jnp.array(32.23)  # LAI correction function parameter
    laim: Union[jax.Array, float] = jnp.array(2.511)  # LAI correction function parameter
    yf: Union[jax.Array, float] = jnp.array(0.5983)  # Fruit harvest coefficient parameter 1
    yl: Union[jax.Array, float] = jnp.array(0.5983)  # Fruit harvest coefficient parameter 2
    z: Union[jax.Array, float] = jnp.array(0.6081)  # Leaf fraction of vegetative dry weight
    phi: Union[jax.Array, float] = jnp.array(4 * (10 ** (-3)))  # heat valve opening
    rh: Union[jax.Array, float] = jnp.array(0.3)  # relative valve opening
    pg: Union[jax.Array, float] = jnp.array(0.475)  # PAR to global radiation ratio
    inj_scale: Union[jax.Array, float] = jnp.array(10 ** (-3))


class DynamicsModel(ABC):
    def __init__(self,
                 dt: float,
                 x_dim: int,
                 u_dim: int,
                 params: PyTree,
                 angle_idx: Optional[Union[int, jax.Array]] = None,
                 dt_integration: float = 0.01,
                 ):
        self.dt = dt
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.params = params
        self.angle_idx = angle_idx

        self.dt_integration = dt_integration
        assert dt >= dt_integration
        assert (dt / dt_integration - int(dt / dt_integration)) < 1e-4, 'dt must be multiple of dt_integration'
        self._num_steps_integrate = int(dt / dt_integration)

    def next_step(self, x: jax.Array, u: jax.Array, params: PyTree) -> jax.Array:
        def body(carry, _):
            q = carry + self.dt_integration * self.ode(carry, u, params)
            return q, None

        next_state, _ = jax.lax.scan(body, x, xs=None, length=self._num_steps_integrate)
        if self.angle_idx is not None:
            theta = next_state[self.angle_idx]
            sin_theta, cos_theta = jnp.sin(theta), jnp.cos(theta)
            next_state = next_state.at[self.angle_idx].set(jnp.arctan2(sin_theta, cos_theta))
        return next_state

    def ode(self, x: jax.Array, u: jax.Array, params) -> jax.Array:
        assert x.shape[-1] == self.x_dim and u.shape[-1] == self.u_dim
        return self._ode(x, u, params)

    @abstractmethod
    def _ode(self, x: jax.Array, u: jax.Array, params) -> jax.Array:
        pass

    def _split_key_like_tree(self, key: jax.random.PRNGKey):
        treedef = jtu.tree_structure(self.params)
        keys = jax.random.split(key, treedef.num_leaves)
        return jtu.tree_unflatten(treedef, keys)

    def sample_params_uniform(self, key: jax.random.PRNGKey, sample_shape: Union[int, Tuple[int]],
                              lower_bound: NamedTuple, upper_bound: NamedTuple):
        keys = self._split_key_like_tree(key)
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        return jtu.tree_map(lambda key, l, u: jax.random.uniform(key, shape=sample_shape + l.shape, minval=l, maxval=u),
                            keys, lower_bound, upper_bound)


class Pendulum(DynamicsModel):
    _metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, dt: float, params: PendulumParams = PendulumParams(), dt_integration: float = 0.005,
                 encode_angle: bool = True):
        super().__init__(dt=dt, x_dim=2, u_dim=1, params=params, angle_idx=0, dt_integration=dt_integration)
        self.encode_angle = encode_angle

        # attributes for rendering
        self.render_mode = 'human'
        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        self.state = None
        self.last_u = None

    def _ode(self, x, u, params: PendulumParams):
        # x represents [theta in rad/s, theta_dot in rad/s^2]
        # u represents [torque]

        x0_dot = x[..., 1]
        # We add drag force: https://www.scirp.org/journal/paperinformation.aspx?paperid=73856
        f_drag_linear = - params.nu * x[..., 1] / (params.m * params.l)
        f_drag_second_order = - params.c_d / params.m * (x[..., 1]) ** 2
        f_drag = f_drag_linear + f_drag_second_order
        x1_dot = params.g / params.l * jnp.sin(x[..., 0]) + u[..., 0] / (params.m * params.l ** 2) + f_drag
        return jnp.stack([x0_dot, x1_dot], axis=-1)

    def next_step(self, x: jax.Array, u: jax.Array, params: PyTree, encode_angle: Optional[bool] = None) -> jax.Array:
        if encode_angle is None:
            encode_angle = self.encode_angle
        if encode_angle:
            assert x.shape[-1] == 3
            theta = jnp.arctan2(x[..., 0], x[..., 1])
            x_radian = jnp.stack([theta, x[..., -1]], axis=-1)
            theta_new, theta_dot_new = jnp.split(super().next_step(x_radian, u, params), 2, axis=-1)
            next_state = jnp.concatenate([jnp.sin(theta_new), jnp.cos(theta_new), theta_dot_new], axis=-1)
            assert next_state.shape == x.shape
            return next_state
        else:
            return super().next_step(x, u, params)

    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(34565)):
        theta_diff = jax.random.uniform(key, shape=(), minval=-0.1, maxval=0.1)
        theta = np.pi + theta_diff
        theta = jnp.arctan2(jnp.sin(theta), jnp.cos(theta))
        self.state = jnp.array([theta, 0.0])
        if self.encode_angle:
            return jnp.array([jnp.sin(theta), jnp.cos(theta), 0.0])
        else:
            return self.state

    def step(self, u: Union[jnp.array, float]):
        self.last_u = u
        self.state = self.next_step(self.state, u, self.params, encode_angle=False)
        if self.encode_angle:
            theta, theta_dot = self.state[..., 0], self.state[..., 1]
            return jnp.stack([jnp.sin(theta), jnp.cos(theta), theta_dot], axis=-1)
        else:
            return self.state

    def render(self):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise RuntimeError("pygame is not installed, run `pip install pygame`")

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_dim, self.screen_dim)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state[0] + jnp.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0] + jnp.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )
        gfxdraw.filled_circle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )

        from gym.envs.classic_control import pendulum
        fname = os.path.join(os.path.dirname(pendulum.__file__), "assets/clockwise.png")
        img = pygame.image.load(fname)
        if self.last_u is not None:
            scale_img = pygame.transform.smoothscale(
                img,
                (scale * abs(float(self.last_u)) / 2, scale * float(abs(self.last_u)) / 2),
            )
            is_flip = bool(self.last_u > 0)
            scale_img = pygame.transform.flip(scale_img, is_flip, True)
            self.surf.blit(
                scale_img,
                (
                    offset - scale_img.get_rect().centerx,
                    offset - scale_img.get_rect().centery,
                ),
            )

        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self._metadata["render_fps"])
            pygame.display.flip()

        else:  # mode == "rgb_array":
            return jnp.transpose(
                jnp.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


class RaceCar(DynamicsModel):
    """
    local_coordinates: bool
        Used to indicate if local or global coordinates shall be used.
        If local, the state x is
            x = [0, 0, theta, vel_r, vel_t, angular_velocity_z]
        else:
            x = [x, y, theta, vel_x, vel_y, angular_velocity_z]
    u = [steering_angle, throttle]
    encode_angle: bool
        Encodes angle to sin and cos if true
    """

    def __init__(self, dt, encode_angle: bool = True, local_coordinates: bool = False, rk_integrator: bool = True):
        self.encode_angle = encode_angle
        x_dim = 6
        super().__init__(dt=dt, x_dim=x_dim, u_dim=2, params=CarParams(), angle_idx=2,
                         dt_integration=1 / 90.)
        self.local_coordinates = local_coordinates
        self.angle_idx = 2
        self.velocity_start_idx = 4 if self.encode_angle else 3
        self.velocity_end_idx = 5 if self.encode_angle else 4
        self.rk_integrator = rk_integrator

    def rk_integration(self, x: jnp.array, u: jnp.array, params: CarParams) -> jnp.array:
        integration_factors = jnp.asarray([self.dt_integration / 2.,
                                           self.dt_integration / 2., self.dt_integration,
                                           self.dt_integration])
        integration_weights = jnp.asarray([self.dt_integration / 6.,
                                           self.dt_integration / 3., self.dt_integration / 3.0,
                                           self.dt_integration / 6.0])

        def body(carry, _):
            """one step of rk integration.
            k_0 = self.ode(x, u)
            k_1 = self.ode(x + self.dt_integration / 2. * k_0, u)
            k_2 = self.ode(x + self.dt_integration / 2. * k_1, u)
            k_3 = self.ode(x + self.dt_integration * k_2, u)

            x_next = x + self.dt_integration * (k_0 / 6. + k_1 / 3. + k_2 / 3. + k_3 / 6.)
            """

            def rk_integrate(carry, ins):
                k = self.ode(carry, u, params)
                carry = carry + k * ins
                outs = k
                return carry, outs

            _, dxs = jax.lax.scan(rk_integrate, carry, xs=integration_factors, length=4)
            dx = (dxs.T * integration_weights).sum(axis=-1)
            q = carry + dx
            return q, None

        next_state, _ = jax.lax.scan(body, x, xs=None, length=self._num_steps_integrate)
        if self.angle_idx is not None:
            theta = next_state[self.angle_idx]
            sin_theta, cos_theta = jnp.sin(theta), jnp.cos(theta)
            next_state = next_state.at[self.angle_idx].set(jnp.arctan2(sin_theta, cos_theta))
        return next_state

    def next_step(self, x: jnp.array, u: jnp.array, params: CarParams) -> jnp.array:
        theta_x = jnp.arctan2(x[..., self.angle_idx], x[..., self.angle_idx + 1]) if self.encode_angle else \
            x[..., self.angle_idx]
        offset = jnp.clip(params.angle_offset, -jnp.pi, jnp.pi)
        theta_x = theta_x + offset
        if not self.local_coordinates:
            # rotate velocity to local frame to compute dx
            velocity_global = x[..., self.velocity_start_idx: self.velocity_end_idx + 1]
            rotated_vel = self.rotate_vector(velocity_global,
                                             -theta_x)
            x = x.at[..., self.velocity_start_idx: self.velocity_end_idx + 1].set(rotated_vel)
        if self.encode_angle:
            x_reduced = self.reduce_x(x)
            if self.rk_integrator:
                x_reduced = self.rk_integration(x_reduced, u, params)
            else:
                x_reduced = super().next_step(x_reduced, u, params)
            next_theta = jnp.atleast_1d(x_reduced[..., self.angle_idx])
            next_x = jnp.concatenate([x_reduced[..., 0:self.angle_idx], jnp.sin(next_theta), jnp.cos(next_theta),
                                      x_reduced[..., self.angle_idx + 1:]], axis=-1)
        else:
            if self.rk_integrator:
                next_x = self.rk_integration(x, u, params)
            else:
                next_x = super().next_step(x, u, params)

        if self.local_coordinates:
            # convert position to local frame
            pos = next_x[..., 0:self.angle_idx] - x[..., 0:self.angle_idx]
            rotated_pos = self.rotate_vector(pos, -theta_x)
            next_x = next_x.at[..., 0:self.angle_idx].set(rotated_pos)
        else:
            # convert velocity to global frame
            new_theta_x = jnp.arctan2(next_x[..., self.angle_idx], next_x[..., self.angle_idx + 1]) \
                if self.encode_angle else next_x[..., self.angle_idx]
            new_theta_x = new_theta_x + offset
            velocity = next_x[..., self.velocity_start_idx: self.velocity_end_idx + 1]
            rotated_vel = self.rotate_vector(velocity, new_theta_x)
            next_x = next_x.at[..., self.velocity_start_idx: self.velocity_end_idx + 1].set(rotated_vel)

        return next_x

    def reduce_x(self, x):
        theta = jnp.arctan2(x[..., self.angle_idx], x[..., self.angle_idx + 1])

        x_reduced = jnp.concatenate([x[..., 0:self.angle_idx], jnp.atleast_1d(theta),
                                     x[..., self.velocity_start_idx:]],
                                    axis=-1)
        return x_reduced

    @staticmethod
    def rotate_vector(v, theta):
        v_x, v_y = v[..., 0], v[..., 1]
        rot_x = v_x * jnp.cos(theta) - v_y * jnp.sin(theta)
        rot_y = v_x * jnp.sin(theta) + v_y * jnp.cos(theta)
        return jnp.concatenate([jnp.atleast_1d(rot_x), jnp.atleast_1d(rot_y)], axis=-1)

    def _accelerations(self, x, u, params: CarParams):
        """Compute acceleration forces for dynamic model.
        Inputs
        -------
        x: jnp.ndarray,
            shape = (6, ) -> [x, y, theta, velocity_r, velocity_t, angular_velocity_z]
        u: jnp.ndarray,
            shape = (2, ) -> [steering_angle, throttle]

        Output
        ------
        acceleration: jnp.ndarray,
            shape = (3, ) -> [a_r, a_t, a_theta]
        """
        i_com = params.i_com
        theta, v_x, v_y, w = x[2], x[3], x[4], x[5]
        m = params.m
        l_f = params.l_f
        l_r = params.l_r
        d_f = params.d_f * params.g
        d_r = params.d_r * params.g
        c_f = params.c_f
        c_r = params.c_r
        b_f = params.b_f
        b_r = params.b_r
        c_m_1 = params.c_m_1
        c_m_2 = params.c_m_2

        c_d = params.c_d

        delta, d = u[0], u[1]

        alpha_f = - jnp.arctan(
            (w * l_f + v_y) /
            (v_x + 1e-6)
        ) + delta
        alpha_r = jnp.arctan(
            (w * l_r - v_y) /
            (v_x + 1e-6)
        )
        f_f_y = d_f * jnp.sin(c_f * jnp.arctan(b_f * alpha_f))
        f_r_y = d_r * jnp.sin(c_r * jnp.arctan(b_r * alpha_r))
        f_r_x = (c_m_1 * d - (c_m_2 ** 2) * v_x - (c_d ** 2) * (v_x * jnp.abs(v_x)))

        v_x_dot = (f_r_x - f_f_y * jnp.sin(delta) + m * v_y * w) / m
        v_y_dot = (f_r_y + f_f_y * jnp.cos(delta) - m * v_x * w) / m
        w_dot = (f_f_y * l_f * jnp.cos(delta) - f_r_y * l_r) / i_com

        acceleration = jnp.array([v_x_dot, v_y_dot, w_dot])
        return acceleration

    def _ode_dyn(self, x, u, params: CarParams):
        """Compute derivative using dynamic model.
        Inputs
        -------
        x: jnp.ndarray,
            shape = (6, ) -> [x, y, theta, velocity_r, velocity_t, angular_velocity_z]
        u: jnp.ndarray,
            shape = (2, ) -> [steering_angle, throttle]

        Output
        ------
        x_dot: jnp.ndarray,
            shape = (6, ) -> time derivative of x

        """
        # state = [p_x, p_y, theta, v_x, v_y, w]. Velocities are in local coordinate frame.
        # Inputs: [\delta, d] -> \delta steering angle and d duty cycle of the electric motor.
        theta, v_x, v_y, w = x[2], x[3], x[4], x[5]
        p_x_dot = v_x * jnp.cos(theta) - v_y * jnp.sin(theta)
        p_y_dot = v_x * jnp.sin(theta) + v_y * jnp.cos(theta)
        theta_dot = w
        p_x_dot = jnp.array([p_x_dot, p_y_dot, theta_dot])

        accelerations = self._accelerations(x, u, params)

        x_dot = jnp.concatenate([p_x_dot, accelerations], axis=-1)
        return x_dot

    def _compute_dx_kin(self, x, u, params: CarParams):
        """Compute kinematics derivative for localized state.
        Inputs
        -----
        x: jnp.ndarray,
            shape = (6, ) -> [x, y, theta, v_x, v_y, w], velocities in local frame
        u: jnp.ndarray,
            shape = (2, ) -> [steering_angle, throttle]

        Output
        ------
        dx_kin: jnp.ndarray,
            shape = (6, ) -> derivative of x

        Assumption: \dot{\delta} = 0.
        """
        p_x, p_y, theta, v_x, v_y, w = x[0], x[1], x[2], x[3], x[4], x[5]  # progress
        m = params.m
        l_f = params.l_f
        l_r = params.l_r
        c_m_1 = params.c_m_1
        c_m_2 = params.c_m_2
        c_d = params.c_d
        delta, d = u[0], u[1]
        v_r = v_x
        v_r_dot = (c_m_1 * d - (c_m_2 ** 2) * v_r - (c_d ** 2) * (v_r * jnp.abs(v_r))) / m
        beta = jnp.arctan(jnp.tan(delta) * 1 / (l_r + l_f))
        v_x_dot = v_r_dot * jnp.cos(beta)
        # Determine accelerations from the kinematic model using FD.
        v_y_dot = (v_r * jnp.sin(beta) * l_r - v_y) / self.dt_integration
        # v_x_dot = (v_r_dot + v_y * w)
        # v_y_dot = - v_x * w
        w_dot = (jnp.sin(beta) * v_r - w) / self.dt_integration
        p_g_x_dot = v_x * jnp.cos(theta) - v_y * jnp.sin(theta)
        p_g_y_dot = v_x * jnp.sin(theta) + v_y * jnp.cos(theta)
        dx_kin = jnp.asarray([p_g_x_dot, p_g_y_dot, w, v_x_dot, v_y_dot, w_dot])
        return dx_kin

    def _compute_dx(self, x, u, params: CarParams):
        """Calculate time derivative of state.
        Inputs:
        ------
        x: jnp.ndarray,
            shape = (6, ) -> [x, y, theta, vel_r, vel_t, angular_velocity_z]
        u: jnp.ndarray,
            shape = (2, ) -> [steering_angle, throttle]
        params: CarParams,

        Output:
        -------
        dx: jnp.ndarray, derivative of x


        If params.use_blend <= 0.5 --> only kinematic model is used, else a blend between nonlinear model
        and kinematic is used.
        """
        use_kin = params.use_blend <= 0.5
        v_x = x[3]
        blend_ratio_ub = jnp.square(params.blend_ratio_ub)
        blend_ratio_lb = jnp.square(params.blend_ratio_lb)
        blend_ratio = (v_x - blend_ratio_ub) / (blend_ratio_lb + 1E-6)
        blend_ratio = blend_ratio.squeeze()
        lambda_blend = jnp.min(jnp.asarray([
            jnp.max(jnp.asarray([blend_ratio, 0])), 1])
        )
        dx_kin_full = self._compute_dx_kin(x, u, params)
        dx_dyn = self._ode_dyn(x=x, u=u, params=params)
        dx_blend = lambda_blend * dx_dyn + (1 - lambda_blend) * dx_kin_full
        dx = (1 - use_kin) * dx_blend + use_kin * dx_kin_full
        return dx

    def _ode(self, x, u, params: CarParams):
        """
        Using kinematic model with blending: https://arxiv.org/pdf/1905.05150.pdf
        Code based on: https://github.com/alexliniger/gym-racecar/

        Inputs:
        ------
        x: jnp.ndarray,
            shape = (6, ) -> [x, y, theta, vel_r, vel_t, angular_velocity_z]
        u: jnp.ndarray,
            shape = (2, ) -> [steering_angle, throttle]
        params: CarParams,

        Output:
        -------
        dx: jnp.ndarray, derivative of x
        """
        delta, d = u[0], u[1]
        delta = jnp.clip(delta, a_min=-1, a_max=1) * params.steering_limit
        d = jnp.clip(d, a_min=-1., a_max=1)  # throttle
        u = u.at[0].set(delta)
        u = u.at[1].set(d)
        dx = self._compute_dx(x, u, params)
        return dx


class SergioDynamics(ABC):
    l_b: float = 0
    lam_lb: float = 0.2
    lam_ub: float = 0.9

    def __init__(self,
                 dt: float,
                 n_cells: int,
                 n_genes: int,
                 params: SergioParams = SergioParams(),
                 dt_integration: float = 0.01,
                 ):
        super().__init__()
        self.dt = dt
        self.n_cells = n_cells
        self.n_genes = n_genes
        self.params = params
        self.x_dim = self.n_cells * self.n_genes

        self.dt_integration = dt_integration
        assert dt >= dt_integration
        assert (dt / dt_integration - int(dt / dt_integration)) < 1e-4, 'dt must be multiple of dt_integration'
        self._num_steps_integrate = int(dt / dt_integration)

    def next_step(self, x: jax.Array, params: PyTree) -> jax.Array:
        def body(carry, _):
            q = carry + self.dt_integration * self.ode(carry, params)
            q = jnp.clip(q, a_min=self.l_b)
            return q, None

        next_state, _ = jax.lax.scan(body, x, xs=None, length=self._num_steps_integrate)
        return next_state

    def ode(self, x: jax.Array, params) -> jax.Array:
        assert x.shape[-1] == self.x_dim
        return self._ode(x, params)

    def production_rate(self, x: jnp.array, params: SergioParams):
        assert x.shape == (self.n_cells, self.n_genes)

        def hill_function(x: jnp.array):
            h = x.mean(0)
            hill_numerator = jnp.power(x, params.power)
            hill_denominator = jnp.power(h, params.power) + hill_numerator
            hill = jnp.where(
                jnp.abs(hill_denominator) < 1e-6, 0, hill_numerator / hill_denominator
            )
            return hill

        hills = hill_function(x)
        hills = hills[:, :, None]
        masked_contribution = params.graph[None] * params.contribution_rates

        # [n_cell_types, n_genes, n_genes]
        # switching mechanism between activation and repression,
        # which is decided via the sign of the contribution rates
        intermediate = jnp.where(
            masked_contribution > 0,
            jnp.abs(masked_contribution) * hills,
            jnp.abs(masked_contribution) * (1 - hills),
        )
        # [n_cell_types, n_genes]
        # sum over regulators, i.e. sum over i in [b, i, j]
        production_rate = params.basal_rates + intermediate.sum(1)
        return production_rate

    def _ode(self, x: jax.Array, params) -> jax.Array:
        assert x.shape == (self.n_cells * self.n_genes,)
        x = x.reshape(self.n_cells, self.n_genes)
        production_rate = self.production_rate(x, params)
        x_next = (production_rate - params.lam * x)
        x_next = x_next.reshape(self.n_cells * self.n_genes)
        return x_next

    def _split_key_like_tree(self, key: jax.random.PRNGKey):
        treedef = jtu.tree_structure(self.params)
        keys = jax.random.split(key, treedef.num_leaves)
        return jtu.tree_unflatten(treedef, keys)

    def sample_single_params(self, key: jax.random.PRNGKey, lower_bound: SergioParams, upper_bound: SergioParams):
        lam_key, contrib_key, basal_key, graph_key, power_key = jax.random.split(key, 5)
        lam = jax.random.uniform(lam_key, shape=(self.n_cells, self.n_genes), minval=lower_bound.lam,
                                 maxval=upper_bound.lam)

        contribution_rates = jax.random.uniform(contrib_key, shape=(self.n_cells,
                                                                    self.n_genes,
                                                                    self.n_genes),
                                                minval=lower_bound.contribution_rates,
                                                maxval=upper_bound.contribution_rates)
        basal_rates = jax.random.uniform(basal_key, shape=(self.n_cells,
                                                           self.n_genes),
                                         minval=lower_bound.basal_rates,
                                         maxval=upper_bound.basal_rates)

        lower_bound_graph = jnp.clip(lower_bound.graph, 0, 2)
        upper_bound_graph = jnp.clip(upper_bound.graph, 0, 2)
        graph = jax.random.randint(graph_key, shape=(self.n_genes, self.n_genes), minval=lower_bound_graph,
                                   maxval=upper_bound_graph) * 1.0
        diag_elements = jnp.diag_indices_from(graph)
        graph = graph.at[diag_elements].set(1.0)
        power = jax.random.uniform(power_key, shape=(1,), minval=lower_bound.power, maxval=upper_bound.power)
        return SergioParams(
            lam=lam,
            contribution_rates=contribution_rates,
            basal_rates=basal_rates,
            power=power,
            graph=graph,
        )

    def sample_params_uniform(self, key: jax.random.PRNGKey, sample_shape: Union[int, Tuple[int]],
                              lower_bound: NamedTuple, upper_bound: NamedTuple):
        if isinstance(sample_shape, int):
            keys = jax.random.split(key, sample_shape)
        else:
            keys = jax.random.split(key, np.prod(sample_shape))
        sampled_params = jax.vmap(self.sample_single_params,
                                  in_axes=(0, None, None))(keys, lower_bound, upper_bound)
        return sampled_params


class GreenHouseDynamics(DynamicsModel):
    state_ub = jnp.array(
        [
            # t_g, t_p, t_s, c_i, v_i, mb, mf, ml, d_p, t_o, t_d, c_o, v_o, w, G, t
            30, 50, 20, 8, 1.0, 10, 150, 15, 1.0, 30.0, 10.0, 0.5, 0.8, 5, 500, 10 ** 6,

        ])

    state_lb = jnp.array(
        [
            0, 0, -10, 0, 0, 0, 0, 0, 0, -5, 8.0, 0, 0.01, 0, 0, 0
        ]
    )

    constraint_lb = jnp.array(
        [
            -273.15, -273.15, -273.15, 0, 0, 0, 0, 0, 0, -273.15, -273.15, 0, 0, 0, 0, 0
        ]
    )

    constraint_ub = jnp.array(
        [
            200, 200, 200, jnp.inf, jnp.inf, jnp.inf, jnp.inf, jnp.inf, 1, 200, 200, jnp.inf, jnp.inf, 10, jnp.inf,
            jnp.inf
        ]
    )

    input_ub = jnp.array([60, 1.0, 1.0, 5.0])
    input_lb = jnp.array([0, 0.0, 0.0, 0.0])
    noise_to = 5
    noise_td = 0.01
    noise_co = 0.001
    noise_vo = 0.1
    noise_w = 1
    noise_g = 20

    noise_std = jnp.array(
        [
            0.05, 0.1, 0.05, 0.01, 0.05, 0.05, 0.1, 0.1, 0.0, noise_to,
            noise_td, noise_co, noise_vo, noise_w, noise_g, 0.0,
        ]
    )

    def __init__(self, use_hf: bool = False, dt: float = 300):

        self.use_hf = use_hf
        self.greenhouse_state_dim = 5
        self.crop_states = 4
        self.exogenous_states = 6
        self.state_dim = self.greenhouse_state_dim + self.crop_states \
                         + self.exogenous_states + 1  # 1 additional state for time
        self.greenhouse_input_dim = 4
        self.eps = 1e-8
        params = GreenHouseParams()
        super().__init__(
            dt=dt,
            x_dim=self.state_dim,
            u_dim=self.greenhouse_input_dim,
            params=params,
            angle_idx=None,
            dt_integration=60,
        )

    def next_step(self, x: jnp.array, u: jnp.array, params: GreenHouseParams) -> jnp.array:
        def body(carry, _):
            q = carry + self.dt_integration * self.ode(carry, u, params)
            q = jnp.clip(q, a_min=self.constraint_lb, a_max=self.constraint_ub)
            return q, None

        x_next, _ = jax.lax.scan(body, x, xs=None, length=self._num_steps_integrate)
        return x_next

    def _ode(self, x, u, params: GreenHouseParams):
        """
        Using kinematic model with blending: https://arxiv.org/pdf/1905.05150.pdf
        Code based on: https://github.com/alexliniger/gym-racecar/

        Inputs:
        ------
        x: jnp.ndarray,
            shape = (6, ) -> [x, y, theta, vel_r, vel_t, angular_velocity_z]
        u: jnp.ndarray,
            shape = (2, ) -> [steering_angle, throttle]
        params: CarParams,

        Output:
        -------
        dx: jnp.ndarray, derivative of x
        """
        if self.use_hf:
            dx = self._greenhouse_dynamics_hf(x, u, params)
        else:
            dx = self._greenhouse_dynamics_lf(x, u, params)
        return dx

    @staticmethod
    def buffer_switching_func(B, b1):
        return 1 - jnp.exp(-b1 * B)

    def get_respiration_param(self, x, u, params: GreenHouseParams):
        ml = x[self.greenhouse_state_dim + 2]
        l_lai = (ml / params.wr) ** (params.laim) / (1 + (ml / params.wr) ** (params.laim))
        R = - params.p1 - params.p5 * l_lai
        return R

    def get_crop_photosynthesis(self, x, u, params: GreenHouseParams):
        t_g, c_i = x[0], x[3]
        ml = x[self.greenhouse_state_dim + 2]
        G = x[-2]
        i_par = params.eta * G * params.mp * params.pg
        c_ppm = (10 ** 6) * params.rg / (params.patm * params.Mco2) * (t_g + params.T0) * c_i
        l_lai = (ml / params.wr) ** (params.laim) / (1 + (ml / params.wr) ** (params.laim))
        p_g = params.pm * l_lai * i_par / (params.p3 + i_par) * c_ppm / (params.p4 + c_ppm)
        return p_g

    def get_harvest_coefficient(self, x, u, params: GreenHouseParams):
        d_p = x[self.greenhouse_state_dim + 3]
        t = x[-1]
        t_g = x[0]
        h = (d_p >= 1) * (params.d1 + params.d2 * jnp.log(t_g / params.d3) - params.d4 * t)
        return h

    def _greenhouse_dynamics_hf(self, x, u, params: GreenHouseParams):
        # C, C, C, m, g/m^3, kg/m^-3
        t_g, t_p, t_s, c_i, v_i = x[0], x[1], x[2], x[3], x[5]
        # g/m^-2, g/m^-2, g/m^-2, []
        mb, mf, ml, d_p = x[self.greenhouse_state_dim], x[self.greenhouse_state_dim + 1], \
            x[self.greenhouse_state_dim + 2], x[self.greenhouse_state_dim + 3]
        t = x[-1]
        egs_start_idx = self.greenhouse_state_dim + self.crop_states
        # C, C, g/m^3, kg/m^-3, m/s, w/m^-2
        t_o, t_d, c_o, v_o, w, G = x[egs_start_idx], x[egs_start_idx + 1], \
            x[egs_start_idx + 2], x[egs_start_idx + 3], x[egs_start_idx + 4], x[egs_start_idx + 5]
        t_h, rwl, rww, phi_c = u[0], u[1], u[2], u[3]
        phi_v = (params.sigma * rwl / (1 + params.chi * rwl) + params.zeta + params.xi * rww) * w \
                + params.psi
        k_v = params.rho_a * params.cp_a * phi_v
        alpha = params.nu * jnp.sqrt(params.tau + jnp.sqrt(jnp.abs(t_g - t_p)))
        s = params.s1 * jnp.power(t_g, 2) + params.s2 * t_g + params.s3
        p_g_star = params.a1 * jnp.exp((params.a2 * t_g) / (params.a3 + t_g + self.eps))
        # convert temp to Kelvin and then from pascal to kpa. v_i is in g/m^3
        p_g = (params.lam * (t_g + params.T0) * v_i)
        Dg = p_g_star - p_g
        # g1: mm/s, g2: [], g4: m^3/g, g3: s m ^2 /micromol
        g = params.g1 * (1 - params.g2 * jnp.exp(-params.g3 * G)) * jnp.exp(-params.g4 * c_i)
        l = params.l1 - params.l2 * t_g
        # gb: mm/s^-1, rho_a = kg/m^-3, cp_a: J/(C kg), Dg: kPa, E:g/(sm^2)
        # G: W/m^2, s: KPa/C
        E = (s * params.eta * G + params.rho_a * params.cp_a * Dg * params.gb) / (l * (s
                                                                                       + params.gamma * (1
                                                                                                         + params.gb / g)
                                                                                       ))
        Wg = params.omega * p_g / (params.patm - p_g + self.eps)
        Wc = params.omega * p_g_star / (params.patm - p_g_star + self.eps)

        t_c = params.epsilon / (params.epsilon + 1) * t_o + 1 / params.epsilon * t_g
        Mc = jax.nn.relu(Wg - Wc) * params.m1 * (jnp.abs(t_g - t_c) ** params.m2)
        dt_g_dt = (k_v + params.kr) * (t_o - t_g) + alpha * (t_p - t_g) + params.ks * (t_s - t_g) \
                  + G * params.eta - l * E + l / (1 + params.epsilon) * Mc
        dt_g_dt = dt_g_dt / params.cg
        # jax.debug.print('l {x}', x=l)
        # jax.debug.print('p_g_star{x}', x=p_g_star)
        # jax.debug.print('t_g {x}', x=t_g)

        phi = params.phi
        rh = params.rh
        phi = 2 * phi * rh / (2 - rh)
        dt_p_dt = params.ap / (params.rho_w * params.cp_w) * (params.beta * G - alpha * (t_p - t_g)) + phi * (
                t_h - t_p)
        dt_p_dt = dt_p_dt / params.vp

        dt_s_dt = 1 / params.cs * (params.ks * (t_g - t_s) + params.kd * (params.Td - t_s))

        dc_i_dt = phi_v * (c_o - c_i) + phi_c * params.inj_scale \
                  + self.get_respiration_param(x, u, params) - params.mu * self.get_crop_photosynthesis(x, u, params)
        dc_i_dt = dc_i_dt / params.vg_ag

        dv_i_dt = (E / 1000 - phi_v * (v_i - v_o) - Mc / 1000) / params.vg_ag

        # Tomato model

        g_f = (params.f1 - params.f2 * d_p) * (params.qg ** ((t_g - params.Tg) / 10.0))
        g_l = g_f * params.v1 * jnp.exp(params.v2 * (t_g - params.v3))
        b = self.buffer_switching_func(mb, params.b1)
        buff_1 = b * (params.f * g_f * mf + params.v * g_l * ml / params.z)
        factor = (params.qr ** ((t_g - params.Tg) / 10.0))
        rf = params.MF * factor
        rl = params.ML * factor
        buff_2 = b * (rf * mf + rl * ml / params.z)
        dmb_dt = self.get_crop_photosynthesis(x, u, params) - buff_1 - buff_2
        h = self.get_harvest_coefficient(x, u, params)
        hf, hl = h * params.yf, h * params.yl
        dmf_dt = (b * g_f - (1 - b) * rf - hf) * mf
        dml_dt = (b * g_l - (1 - b) * rl - hl) * ml
        dd_p_dt = params.d1 + params.d2 * jnp.log(t_g / params.d3) - params.d4 * t - h

        # Exogenous effects

        dt_o_dt = jnp.zeros_like(dt_g_dt)
        dt_d_dt = jnp.zeros_like(dt_g_dt)
        dc_o_dt = jnp.zeros_like(dt_g_dt)
        dv_o_dt = jnp.zeros_like(dt_g_dt)
        dw_dt = jnp.zeros_like(dt_g_dt)
        dG_dt = jnp.zeros_like(dt_g_dt)

        # Time
        dt_dt = jnp.ones_like(dt_g_dt)

        dx_dt = jnp.stack([
            dt_g_dt, dt_p_dt, dt_s_dt, dc_i_dt, dv_i_dt,
            dmb_dt, dmf_dt, dml_dt, dd_p_dt,
            dt_o_dt, dt_d_dt, dc_o_dt, dv_o_dt, dw_dt, dG_dt, dt_dt,
        ])
        return dx_dt

    def _greenhouse_dynamics_lf(self, x, u, params: GreenHouseParams):

        t_g, t_p, t_s, c_i, v_i = x[0], x[1], x[2], x[3], x[5]
        # g/m^-2, g/m^-2, g/m^-2, []
        mb, mf, ml, d_p = x[self.greenhouse_state_dim], x[self.greenhouse_state_dim + 1], \
            x[self.greenhouse_state_dim + 2], x[self.greenhouse_state_dim + 3]
        t = x[-1]
        egs_start_idx = self.greenhouse_state_dim + self.crop_states
        # C, C, g/m^3, kg/m^-3, m/s, w/m^-2
        t_o, t_d, c_o, v_o, w, G = x[egs_start_idx], x[egs_start_idx + 1], \
            x[egs_start_idx + 2], x[egs_start_idx + 3], x[egs_start_idx + 4], x[egs_start_idx + 5]
        t_h, rwl, rww, phi_c = u[0], u[1], u[2], u[3]
        phi_v = (params.sigma * rwl / (1 + params.chi * rwl) + params.zeta + params.xi * rww) * w \
                + params.psi
        k_v = params.rho_a * params.cp_a * phi_v

        dt_g_dt = (k_v + params.kr) * (t_o - t_g) + params.ks * (t_s - t_g) \
                  + G * params.eta
        dt_g_dt = dt_g_dt / params.cg
        # jax.debug.print('l {x}', x=l)
        # jax.debug.print('p_g_star{x}', x=p_g_star)
        # jax.debug.print('t_g {x}', x=t_g)

        dt_p_dt = jnp.zeros_like(dt_g_dt)

        dt_s_dt = 1 / params.cs * (params.ks * (t_g - t_s) + params.kd * (params.Td - t_s))

        dc_i_dt = phi_v * (c_o - c_i) + phi_c * params.inj_scale \
                  + self.get_respiration_param(x, u, params) - params.mu * self.get_crop_photosynthesis(x, u, params)
        dc_i_dt = dc_i_dt / params.vg_ag

        dv_i_dt = jnp.zeros_like(dt_g_dt)

        # Tomato model

        g_f = (params.f1 - params.f2 * d_p) * (params.qg ** ((t_g - params.Tg) / 10.0))
        g_l = g_f * params.v1 * jnp.exp(params.v2 * (t_g - params.v3))
        b = self.buffer_switching_func(mb, params.b1)
        buff_1 = b * (params.f * g_f * mf + params.v * g_l * ml / params.z)
        factor = (params.qr ** ((t_g - params.Tg) / 10.0))
        rf = params.MF * factor
        rl = params.ML * factor
        buff_2 = b * (rf * mf + rl * ml / params.z)
        dmb_dt = self.get_crop_photosynthesis(x, u, params) - buff_1 - buff_2
        h = self.get_harvest_coefficient(x, u, params)
        hf, hl = h * params.yf, h * params.yl
        dmf_dt = (b * g_f - (1 - b) * rf - hf) * mf
        dml_dt = (b * g_l - (1 - b) * rl - hl) * ml
        dd_p_dt = params.d1 + params.d2 * jnp.log(t_g / params.d3) - params.d4 * t - h

        # Exogenous effects

        dt_o_dt = jnp.zeros_like(dt_g_dt)
        dt_d_dt = jnp.zeros_like(dt_g_dt)
        dc_o_dt = jnp.zeros_like(dt_g_dt)
        dv_o_dt = jnp.zeros_like(dt_g_dt)
        dw_dt = jnp.zeros_like(dt_g_dt)
        dG_dt = jnp.zeros_like(dt_g_dt)

        # Time
        dt_dt = jnp.ones_like(dt_g_dt)

        dx_dt = jnp.stack([
            dt_g_dt, dt_p_dt, dt_s_dt, dc_i_dt, dv_i_dt,
            dmb_dt, dmf_dt, dml_dt, dd_p_dt,
            dt_o_dt, dt_d_dt, dc_o_dt, dv_o_dt, dw_dt, dG_dt, dt_dt,
        ])

        return dx_dt

    def sample_single_params(self, key: jax.random.PRNGKey,
                             lower_bound: NamedTuple, upper_bound: NamedTuple):
        keys = self._split_key_like_tree(key)
        return jtu.tree_map(lambda key, l, u: jax.random.uniform(key, shape=l.shape, minval=l, maxval=u),
                            keys, lower_bound, upper_bound)


if __name__ == "__main__":
    dim_x, dim_y = 10, 10
    sim = SergioDynamics(0.1, dim_x, dim_y)
    x_next = sim.next_step(x=jnp.ones(dim_x * dim_y), params=sim.params)
    lower_bound = SergioParams(lam=jnp.array(0.2),
                               contribution_rates=jnp.array(1.0),
                               basal_rates=jnp.array(1.0),
                               graph=jnp.array(0))
    upper_bound = SergioParams(lam=jnp.array(0.9),
                               contribution_rates=jnp.array(5.0),
                               basal_rates=jnp.array(5.0),
                               graph=jnp.array(2))
    key = jax.random.PRNGKey(0)
    keys = random.split(key, 4)
    params = vmap(sim.sample_params_uniform, in_axes=(0, None, None, None))(keys, 1, lower_bound, upper_bound)
    x_next = vmap(vmap(lambda p: sim.next_step(x=jnp.ones(dim_x * dim_y), params=p)))(params)
    pendulum = Pendulum(0.1)
    pendulum.next_step(x=jnp.array([0., 0., 0.]), u=jnp.array([1.0]), params=pendulum.params)

    upper_bound = PendulumParams(m=jnp.array(1.0), l=jnp.array(1.0), g=jnp.array(10.0), nu=jnp.array(1.0),
                                 c_d=jnp.array(1.0))
    lower_bound = PendulumParams(m=jnp.array(0.1), l=jnp.array(0.1), g=jnp.array(9.0), nu=jnp.array(0.1),
                                 c_d=jnp.array(0.1))
    key = jax.random.PRNGKey(0)
    keys = random.split(key, 4)
    params = vmap(pendulum.sample_params_uniform, in_axes=(0, None, None, None))(keys, 1, lower_bound, upper_bound)


    def simulate_car(init_pos=jnp.zeros(2), horizon=150):
        dt = 0.1
        car = RaceCar(dt=dt, encode_angle=False)
        params = CarParams(use_blend=0.0)
        x = jnp.zeros(6)
        x_traj = jnp.zeros([horizon, 2])
        x = x.at[0:2].set(init_pos)
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for h in range(horizon):
            s = 0.35
            d = 1
            u = jnp.asarray([s, d])
            x = car.next_step(x, u, params)
            x_traj = x_traj.at[h, ...].set(x[0:2])

        ax.plot(x_traj[:, 0], x_traj[:, 1], label='Car Trajectory w/o blend')

        params = CarParams(use_blend=1.0)
        x = jnp.zeros(6)
        x_traj = jnp.zeros([horizon, 2])
        x = x.at[0:2].set(init_pos)
        for h in range(horizon):
            s = 0.35
            d = 1
            u = jnp.asarray([s, d])
            x = car.next_step(x, u, params)
            x_traj = x_traj.at[h, ...].set(x[0:2])

        ax.plot(x_traj[:, 0], x_traj[:, 1], label='Car Trajectory with blend')
        # plt.scatter(env._goal[0], env._goal[1], color='red', label='goal')
        plt.legend()
        plt.xlabel('x-distance in [m]')
        plt.ylabel('y-distance in [m]')
        plt.title("Simulation of Car for " + str(int(horizon)))
        plt.show()


    simulate_car()
