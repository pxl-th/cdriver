from itertools import product, combinations
from math import atan2
from os.path import join
from typing import Tuple

from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
from matplotlib.animation import FuncAnimation
from numpy import (
    ndarray, array, pi, zeros, sum, dot, diag, sin, cos, tan, radians, loadtxt,
    hstack,
)
from numpy.random import randn
from pymap3d import geodetic2enu, Ellipsoid
from scipy.spatial.transform import Rotation

from tqdm import tqdm


def gnss2enu(gnss: ndarray) -> ndarray:
    e, n, u = geodetic2enu(
        gnss[:, 0], gnss[:, 1], gnss[:, 2],
        gnss[0, 0], gnss[0, 1], gnss[0, 2],
        ell=Ellipsoid(model="wgs84"),
    )
    return hstack((e.reshape(-1, 1), n.reshape(-1, 1), u.reshape(-1, 1)))


def fx(x: ndarray, dt: float, u: ndarray = (0, 0), wheelbase: float = 0):
    """
    Kalman:
        - state: [x (m), y (m), theta (rad)]; theta -- orientation (compass)
        - control: [v (m/s), alpha (rad)]; alpha -- steering angle
        - measurement: [x (m), y (m)]; gnss position
    """
    velocity, steering_angle = u
    if abs(velocity) < 0.3:  # less than 1 km/h
        return x

    heading = x[2]
    distance = velocity * dt

    if abs(steering_angle) < 0.017:  # less than 1 degree
        return x + array([
            distance * cos(heading), distance * sin(heading), 0,
        ])

    beta = (distance / wheelbase) * tan(steering_angle)
    radius = distance / beta
    return x + array([
        -radius * sin(heading) + radius * sin(heading + beta),
        radius * cos(heading) - radius * cos(heading + beta),
        beta,
    ])


def normalize_angle(angle):
    angle = angle % (2 * pi)
    if angle > pi:
        angle -= 2 * pi
    return angle


def residual_x(a, b):
    y = a - b
    y[2] = normalize_angle(y[2])
    return y


def hx_gnss(x):
    return x.copy()


def state_mean(sigmas, Wm):
    """
    Compute mean over states
    """
    x = zeros(3)

    sum_sin = sum(dot(sin(sigmas[:, 2]), Wm))
    sum_cos = sum(dot(cos(sigmas[:, 2]), Wm))

    x[0] = sum(dot(sigmas[:, 0], Wm))
    x[1] = sum(dot(sigmas[:, 1], Wm))
    x[2] = atan2(sum_sin, sum_cos)
    return x


def fuse_sensors(
    ukf: UnscentedKalmanFilter,
    positions: ndarray, positions_timestamps: ndarray,
    speed: ndarray, steer: ndarray, compass: ndarray, timestamps: ndarray,
    wheelbase: float, dt: float
):
    r"""
    Fuse GNSS data, which has lower frequency
    with (speed, steering, compass) with higher frequency
    to estimate position and rotation of a vehicle
    more frequently and accurately than by GNSS alone.

    For example GNSS has 5 Hz frequency, and other sensors --- 50 Hz.
    This assumes, that (speed, steer, compass) are synchronized.

    Args:
        ukf (UnscentedKalmanFilter):
            Unscented Kalman Filter instance.
        positions ((N, 2) ndarray[float]):
            (x, y) positions from GNSS in ENU corrdinates (in meters).
        positions_timestamps ((N,) ndarray[float]):
            Timestamps of each position.
        speed ((M,) ndarray[float]):
            Speed in m/s of a vehicle.
        steer ((M,) ndarray[float]):
            Steering angles of a vehicle in *radians*.
            Positive values --- clockwise,
            negative values --- counter-clockwise steering.
            Steering angles should be in
            `[-max_steering_angle, max_steering_angle]` range.
        compass ((M,) ndarray[float]):
            Compass angles in *radians*.
        timestamps ((M,) ndarray[float]):
            Timestamps of (speed, steer, compass) data.
        wheelbase (float):
            Length of a wheelbase in meters.
        dt (float):
            Inverse of frequency of (speed, steer, compass) sensors.

    *Note*: `N \leq M`.
    """
    track = [ukf.x]
    pos_i = 0

    for i, t in enumerate(timestamps):
        ukf.predict(u=array([speed[i], steer[i]]), wheelbase=wheelbase)

        can_update = pos_i < positions.shape[0] and (
            abs(t - positions_timestamps[pos_i]) < dt
            or t > positions_timestamps[pos_i]
        )
        if can_update:
            measurement = array([
                positions[pos_i, 0], positions[pos_i, 1],
                compass[i],
            ])
            ukf.update(z=measurement)
            pos_i += 1
        track.append(ukf.x)
    return array(track)


# Visualization of fusion
def get_cube_vertices(span: Tuple[float, float]) -> ndarray:
    cube = []
    for s, e in combinations(array(list(product(span, span, span))), 2):
        if sum(abs(s - e)) == span[1] - span[0]:
            cube.append((s, e))
    return array(cube)


def update_cube(i, axes, cube, track, bar):
    bar.update()

    axes.clear()

    cshape = cube.shape
    rotation = Rotation.from_euler("z", track[i, 2], degrees=False)
    cube = rotation.apply(cube.reshape(-1, 3)).reshape(*cshape)
    cube[:, :, 0] += track[i, 0]
    cube[:, :, 1] += track[i, 1]

    span = 50
    axes.set_xlim(track[i, 0] - span, track[i, 0] + span)
    axes.set_ylim(track[i, 1] - span, track[i, 1] + span)
    axes.set_zlim(0, span)
    axes.set_axis_off()

    if i > 1:
        axes.plot3D(
            track[:i, 0], track[:i, 1], zeros(i) + 1,
            color="red", linewidth=1, linestyle="dashed",
        )
    for edge in cube:
        axes.plot3D(
            edge[:, 0], edge[:, 1], edge[:, 2],
            color="black", linewidth=2, linestyle="solid",
        )


def visualize_track(track: ndarray, save_path: str):
    figure = pyplot.figure()
    ax = figure.add_axes([0, 0, 1, 1], projection="3d")

    cube = get_cube_vertices([-1, 1])
    cube[:, :, 0] *= 2

    bar = tqdm(total=track.shape[0])
    animation = FuncAnimation(
        figure, update_cube,
        fargs=(ax, cube, track, bar),
        interval=1, frames=track.shape[0],
    )
    animation.save(
        filename=save_path, fps=60, extra_args=["-vcodec", "libx264"],
    )


def gnss_fusion():
    """
    Example of usage
    """
    base = (
        r"C:\Users\tonys\projects\carla-dataset"
        r"\recording-2020-04-28-13-26-24-397935"
    )
    gnss_time = loadtxt(join(base, "gnss", "timestamp"), dtype="float32")
    speed_time = loadtxt(join(base, "imu", "timestamp"), dtype="float32")

    gnss = loadtxt(join(base, "gnss", "gnss"), dtype="float32")
    speed = loadtxt(join(base, "imu", "speed"), dtype="float32")
    steer = loadtxt(join(base, "imu", "steer"), dtype="float32")
    compass = loadtxt(join(base, "imu", "compass"), dtype="float32")

    dt = 1 / 50
    wheelbase = 2.0
    sigma_pos = 1.5

    steer_limit = 70
    steer *= -steer_limit
    steer = radians(steer)

    compass = radians(compass)
    positions = gnss2enu(gnss)[:, :2]
    noisy_positions = positions + randn(*positions.shape) * sigma_pos

    points = MerweScaledSigmaPoints(
        n=3, alpha=1e-3, beta=2.0, kappa=0, subtract=residual_x,
    )
    ukf = UnscentedKalmanFilter(
        dim_x=3, dim_z=3, dt=dt,
        hx=hx_gnss, fx=fx, points=points,
        x_mean_fn=state_mean, residual_x=residual_x, residual_z=residual_x,
    )
    ukf.x = array([positions[0, 0], positions[0, 0], compass[0]])
    ukf.P = diag([0.1, 0.1, 0.1])  # initial variance
    ukf.R = diag([0.1, 0.1, 0.1])  # measurement variance
    ukf.Q = Q_discrete_white_noise(3, dt=dt, var=0.0001)

    track = fuse_sensors(
        ukf,
        noisy_positions, gnss_time,
        speed, steer, compass, speed_time,
        wheelbase, dt,
    )

    # save_path = (
    #     r"C:\Users\tonys\projects\CARLA\PythonAPI\examples\cdriver\fusion"
    #     r"\orientation.mp4"
    # )
    # visualize_track(track, save_path)

    pyplot.scatter(
        noisy_positions[:, 0], noisy_positions[:, 1],
        label="Noisy GNSS", s=1,
    )
    pyplot.plot(track[:, 0], track[:, 1], label="Predicted", c="red")
    pyplot.title("ENU: East - right, North - up")
    pyplot.legend()
    pyplot.show()


if __name__ == "__main__":
    gnss_fusion()
