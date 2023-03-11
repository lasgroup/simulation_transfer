from abc import ABC, abstractmethod
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random, vmap
from jaxtyping import PyTree


class PendulumParams(NamedTuple):
    m: jax.Array = jnp.array(0.15)
    l: jax.Array = jnp.array(0.5)
    g: jax.Array = jnp.array(9.81)
    nu: jax.Array = jnp.array(0.1)
    c_d: jax.Array = jnp.array(0.1)


class CarParams(NamedTuple):
    """
    Range taken from: https://www.jstor.org/stable/pdf/44470677.pdf
    d_f, d_r : Represent grip of the car -> High grip means d_f, d_r = 1.0. Low grip d_f, d_r ~ 0.0,
                Typically 0.8 - 0.9
    b_f, b_r: Slope of the pacejka. Typically, between [0.5 - 2.5].

    delta_limit: [0.3 - 0.5] -> Limit of the steering angle.

    c_f, c_r: [1.0 2.0] # motor parameters: source https://web.mit.edu/drela/Public/web/qprop/motor1_theory.pdf,
    https://ethz.ch/content/dam/ethz/special-interest/mavt/dynamic-systems-n-control/idsc-dam/Lectures/Embedded
    -Control-Systems/LectureNotes/6_Motor_Control.pdf # or look at:
    https://video.ethz.ch/lectures/d-mavt/2021/spring/151-0593-00L/00718f4f-116b-4645-91da-b9482164a3c7.html :
    lecture 2 part 2
    c_m_1: max current of motor: [0.2 - 0.5] c_m_2: motor resistance due to shaft: [0.01 - 0.15]

    c_rr: zero torque current: [0 0.1]

    tv_p: [0.0 0.1]

    """
    m: jax.Array = jnp.array(0.05)
    l: jax.Array = jnp.array(0.06)
    a: jax.Array = jnp.array(0.25)
    b: jax.Array = jnp.array(0.01)
    g: jax.Array = jnp.array(9.81)
    d_f: jax.Array = jnp.array(0.2)
    c_f: jax.Array = jnp.array(1.25)
    b_f: jax.Array = jnp.array(2.5)
    d_r: jax.Array = jnp.array(0.2)
    c_r: jax.Array = jnp.array(1.25)
    b_r: jax.Array = jnp.array(2.5)
    c_m_1: jax.Array = jnp.array(0.2)
    c_m_2: jax.Array = jnp.array(0.05)
    c_rr: jax.Array = jnp.array(0.0)  # motor friction
    c_d_max: jax.Array = jnp.array(0.1)
    c_d_min: jax.Array = jnp.array(0.01)
    delta_limit: jax.Array = jnp.array(0.5)
    tv_p: jax.Array = jnp.array(0.0)


class DynamicsModel(ABC):
    def __init__(self, h: float, x_dim: int, u_dim: int, params_example: PyTree):
        self.dt = h
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.params_example = params_example

    def next_step(self, x: jax.Array, u: jax.Array, params: PyTree) -> jax.Array:
        return x + self.dt * self.ode(x, u, params)

    def ode(self, x: jax.Array, u: jax.Array, params) -> jax.Array:
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,)
        return self._ode(x, u, params)

    @abstractmethod
    def _ode(self, x: jax.Array, u: jax.Array, params) -> jax.Array:
        pass

    def random_split_like_tree(self, key):
        treedef = jtu.tree_structure(self.params_example)
        keys = jax.random.split(key, treedef.num_leaves)
        return jtu.tree_unflatten(treedef, keys)

    def sample_params(self, key, upper_bound, lower_bound):
        keys = self.random_split_like_tree(key)
        return jtu.tree_map(
            lambda key, l_bound, u_bound: jax.random.uniform(key, shape=l_bound.shape, minval=l_bound,
                                                             maxval=u_bound), keys, lower_bound, upper_bound)


class Pendulum(DynamicsModel):
    def __init__(self, h):
        super().__init__(h=h, x_dim=2, u_dim=1, params_example=PendulumParams())

    def _ode(self, x, u, params: PendulumParams):
        # x represents [theta in rad/s, theta_dot in rad/s^2]
        # u represents [torque]

        x0_dot = x[1]
        # We add drag force: https://www.scirp.org/journal/paperinformation.aspx?paperid=73856
        f_drag_linear = - params.nu * x[1] / (params.m * params.l)
        f_drag_second_order = - params.c_d / params.m * (x[1]) ** 2
        f_drag = f_drag_linear + f_drag_second_order
        x1_dot = params.g / params.l * jnp.sin(x[0]) + + u[0] \
                 / (params.m * params.l ** 2) + f_drag
        return jnp.array([x0_dot, x1_dot])


class BicycleModel(DynamicsModel):

    def _accelerations(self, x, u, params: CarParams):
        i_com = self._get_moment_of_intertia(params)
        theta, v_x, v_y, w = x[2], x[3], x[4], x[5]
        m = params.m
        l = params.l
        d_f = params.d_f * params.g * m
        d_r = params.d_r * params.g * m
        c_f = params.c_f
        c_r = params.c_r
        b_f = params.b_f
        b_r = params.b_r
        c_m_1 = params.c_m_1
        c_m_2 = params.c_m_2
        c_d_max = params.c_d_max
        c_d_min = params.c_d_min
        c_rr = params.c_rr
        a = params.a
        l_r = self._get_x_com(params)
        l_f = l - l_r
        tv_p = params.tv_p

        c_d = c_d_min + (c_d_max - c_d_min) * a

        delta, d = u[0], u[1]
        delta = jnp.clip(delta, a_min=-params.delta_limit,
                         a_max=params.delta_limit)
        d = jnp.clip(d, a_min=-1, a_max=1)

        w_tar = delta * v_x/l

        alpha_f = -jnp.arctan(
            (w * l_f + v_y)/
            (v_x + 1e-6)
        ) + delta
        alpha_r = jnp.arctan(
            (w * l_r - v_y)/
            (v_x + 1e-6)
        )
        f_f_y = d_f * jnp.sin(c_f * jnp.arctan(b_f * alpha_f))
        f_r_y = d_r * jnp.sin(c_r * jnp.arctan(b_r * alpha_r))
        f_r_x = (c_m_1 - c_m_2 * v_x) * d - c_rr - c_d * (v_x ** 2)

        v_x_dot = (f_r_x - f_f_y * jnp.sin(delta) + m * v_y * w) / m
        v_y_dot = (f_r_y + f_f_y * jnp.cos(delta) - m * v_x * w) / m
        w_dot = (f_f_y * l_f * jnp.cos(delta) - f_r_y * l_r + tv_p * (w_tar - w)) / i_com

        acceleration = jnp.array([v_x_dot, v_y_dot, w_dot])
        return acceleration

    @staticmethod
    def _get_x_com(params: CarParams):
        x_com = params.l * (params.a + 2) / (3 * (params.a + 1))
        return x_com

    def _get_moment_of_intertia(self, params: CarParams):
        # Moment of inertia around origin
        a = params.a
        assert (0 < a <= 1), "a must be between 0 and 1."
        b = params.b
        m = params.m
        l = params.l
        i_o = m / (6 * (1 + a)) * ((a ** 3 + a ** 2 + a + 1) * (b ** 2) + (l ** 2) * (a + 3))
        x_com = self._get_x_com(params)
        i_com = i_o - params.m * (x_com ** 2)
        return i_com

    def _ode_dyn(self, x, u, params: CarParams):
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

    def _ode_kin(self, x, u, params: CarParams):
        p_x, p_y, theta, v_x = x[0], x[1], x[2], x[3]  # progress
        m = params.m
        l = params.l
        c_m_1 = params.c_m_1
        c_m_2 = params.c_m_2
        c_d_max = params.c_d_max
        c_d_min = params.c_d_min
        c_rr = params.c_rr
        a = params.a
        l_r = self._get_x_com(params)

        c_d = c_d_min + (c_d_max - c_d_min) * a

        delta, d = u[0], u[1]

        d_0 = (c_rr + c_d * (v_x**2))/(c_m_1 - c_m_2 * v_x)
        d_slow = jnp.maximum(d, d_0)
        d_fast = d

        slow_ind = v_x <= 0.1
        d_applied = d_slow * slow_ind + d_fast * (~slow_ind)
        f_r_x = ((c_m_1 - c_m_2 * v_x) * d_applied - c_rr - c_d * (v_x ** 2))/m

        beta = jnp.arctan(l_r * jnp.arctan(delta) / l)
        p_x_dot = v_x * jnp.cos(beta + theta)  # s_dot
        p_y_dot = v_x * jnp.sin(beta + theta)  # d_dot
        w = v_x * jnp.sin(beta) / l_r

        dx_kin = jnp.asarray([p_x_dot, p_y_dot, w, f_r_x])
        return dx_kin

    def _ode(self, x, u, params: CarParams):
        """
        Using kinematic model with blending: https://arxiv.org/pdf/1905.05150.pdf
        Code based on: https://github.com/manish-pra/copg/blob/4a370594ab35f000b7b43b1533bd739f70139e4e/car_racing_simulator/VehicleModel.py#L381
        """
        v_x = x[3]
        blend_ratio = (v_x - 0.3) / (0.2)
        l = params.l
        l_r = self._get_x_com(params)

        lambda_blend = jnp.min([jnp.max([blend_ratio, 0]), 1])

        if lambda_blend < 1:
            v_x = x[3]
            v_y = x[4]
            x_kin = jnp.asarray([x[0], x[1], x[2], jnp.sqrt(v_x**2 + v_y**2)])
            dxkin = self._ode_kin(x_kin, u, params)
            delta = u[0]
            beta = jnp.arctan(l_r * jnp.tan(delta) / l)
            v_x_state = dxkin[3] * jnp.cos(beta)  # V*cos(beta)
            v_y_state = dxkin[3] * jnp.sin(beta)  # V*sin(beta)
            w = v_x_state * jnp.arctan(delta) / l

            x_kin_full = jnp.asarray([dxkin[0], dxkin[1], dxkin[2], v_x_state, v_y_state, w])

            if lambda_blend == 0:
                return x_kin_full

        if lambda_blend > 0:
            dxdyn = self._ode_dyn(x=x, u=u, params=params)

            if lambda_blend == 1:
                return dxdyn

        return lambda_blend * dxdyn + (1-lambda_blend)*dxkin



if __name__ == "__main__":
    pendulum = Pendulum(0.1)
    upper_bound = PendulumParams(m=jnp.array(1.0), l=jnp.array(1.0), g=jnp.array(10.0), nu=jnp.array(1.0),
                                 c_d=jnp.array(1.0))
    lower_bound = PendulumParams(m=jnp.array(0.1), l=jnp.array(0.1), g=jnp.array(9.0), nu=jnp.array(0.1),
                                 c_d=jnp.array(0.1))
    key = jax.random.PRNGKey(0)
    keys = random.split(key, 4)
    params = vmap(pendulum.sample_params, in_axes=(0, None, None))(keys, upper_bound, lower_bound)
