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
    u_b: float = 500
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
            return q, None

        next_state, _ = jax.lax.scan(body, x, xs=None, length=self._num_steps_integrate)
        return next_state

    def ode(self, x: jax.Array, params) -> jax.Array:
        assert x.shape[-1] == self.x_dim
        return self._ode(x, params)

    def production_rate(self, x: jnp.array, params: SergioParams):
        assert x.shape == (self.n_cells, self.n_genes)

        def hill_function(x: jnp.array, n: float = 2.0):
            h = x.mean(0)
            hill_numerator = jnp.power(x, n)
            hill_denominator = jnp.power(h, n) + hill_numerator
            hill = jnp.where(
                jnp.abs(hill_denominator) < 1e-6, 0, hill_numerator / hill_denominator
            )
            return hill

        hills = hill_function(x)
        masked_contribution = params.contribution_rates

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
        x_next = (production_rate - params.lam * x) * self.dt
        x_next = x_next.reshape(self.n_cells * self.n_genes)
        x_next = jnp.clip(x_next, self.l_b, self.u_b)
        return x_next

    def _split_key_like_tree(self, key: jax.random.PRNGKey):
        treedef = jtu.tree_structure(self.params)
        keys = jax.random.split(key, treedef.num_leaves)
        return jtu.tree_unflatten(treedef, keys)

    def sample_single_params(self, key: jax.random.PRNGKey, lower_bound: NamedTuple, upper_bound: NamedTuple):
        lam_key, contrib_key, basal_key = jax.random.split(key, 3)
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

        return SergioParams(
            lam=lam,
            contribution_rates=contribution_rates,
            basal_rates=basal_rates
        )

    def sample_params_uniform(self, key: jax.random.PRNGKey, sample_shape: Union[int, Tuple[int]],
                              lower_bound: NamedTuple, upper_bound: NamedTuple):
        if isinstance(sample_shape, int):
            keys = jax.random.split(key, sample_shape)
        else:
            keys = jax.random.split(key, jnp.prod(sample_shape.shape))
        sampled_params = jax.vmap(self.sample_single_params,
                                  in_axes=(0, None, None))(keys, lower_bound, upper_bound)
        return sampled_params


if __name__ == "__main__":
    sim = SergioDynamics(0.1, 20, 20)
    x_next = sim.next_step(x=jnp.ones(20 * 20), params=sim.params)
    lower_bound = SergioParams(lam=jnp.array(0.2),
                               contribution_rates=jnp.array(1.0),
                               basal_rates=jnp.array(1.0))
    upper_bound = SergioParams(lam=jnp.array(0.9),
                               contribution_rates=jnp.array(5.0),
                               basal_rates=jnp.array(5.0))
    key = jax.random.PRNGKey(0)
    keys = random.split(key, 4)
    params = vmap(sim.sample_params_uniform, in_axes=(0, None, None, None))(keys, 1, lower_bound, upper_bound)
    x_next = vmap(vmap(lambda p: sim.next_step(x=jnp.ones(20 * 20), params=p)))(params)
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
