from os.path import join
from matplotlib import pyplot
from numpy import array, loadtxt, hstack, ndarray, tile, zeros, eye
from numpy.random import randn

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

from pymap3d import geodetic2enu, Ellipsoid

from ahrs.filters import Madgwick
from ahrs.common import DEG2RAD, Quaternion


def gnss2enu(gnss: ndarray) -> ndarray:
    e, n, u = geodetic2enu(
        gnss[:, 0], gnss[:, 1], gnss[:, 2],
        gnss[0, 0], gnss[0, 1], gnss[0, 2],
        ell=Ellipsoid(model="wgs84"),
    )
    return hstack((e.reshape(-1, 1), n.reshape(-1, 1), u.reshape(-1, 1)))


def imu(acceleration: ndarray, gyroscope: ndarray) -> ndarray:
    madgwick = Madgwick(beta=0.1, frequency=50.0)
    quaternions = tile(
        array([1, 0, 0, 0], dtype="float32"),
        (acceleration.shape[0], 1),
    )
    rotations = zeros((acceleration.shape[0], 3, 3), dtype="float32")
    for t in range(1, acceleration.shape[0]):
        quaternions[t] = madgwick.updateIMU(
            quaternions[t - 1], DEG2RAD * gyroscope[t], acceleration[t],
        )
        rotations[t] = Quaternion(quaternions[t]).to_DCM()
    return rotations[1:]


def localize():
    base = (
        r"C:\Users\tonys\projects\carla-dataset"
        r"\recording-2020-04-25-22-54-06-705312"
    )
    gnss_time = loadtxt(join(base, "gnss", "timestamp"), dtype="float32")
    acc_time = loadtxt(join(base, "imu", "timestamp"), dtype="float32")[1:]

    gnss = loadtxt(join(base, "gnss", "gnss"), dtype="float32")  # lat, lon, alt
    accelerations = loadtxt(join(base, "imu", "accelerometer"), dtype="float32")
    gyroscope = loadtxt(join(base, "imu", "gyro"), dtype="float32")

    rotations = imu(accelerations, gyroscope)
    accelerations = accelerations[1:]
    absolute_accelerations = zeros(accelerations.shape, dtype="float32")
    for i, (rotation, acceleration) in enumerate(zip(rotations, accelerations)):
        absolute_accelerations[i] = rotation.T.dot(acceleration)

    # pyplot.plot(accelerations[:, 0])
    # pyplot.plot(accelerations[:, 1])
    pyplot.plot(absolute_accelerations[:, 0])
    pyplot.plot(absolute_accelerations[:, 1])
    pyplot.show()

    positions = gnss2enu(gnss)[:, :2]
    absolute_accelerations = absolute_accelerations[:, :2]

    ground_truth_noise = 0.5
    process_variance = 1.0
    position_variance = 0.001  # more like how much we trust measurement
    acceleration_variance = 1.0

    noised_positions = positions + randn(*positions.shape) * ground_truth_noise
    corrected_positions = zeros(absolute_accelerations.shape, dtype="float32")
    dt = 1 / 50
    # measurement function for position and acceleration
    H_p = array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]], dtype="float32")
    H_a = array([[0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1]], dtype="float32")
    # measurement noise matrix for position and acceleration
    R_p = eye(2, dtype="float32") * (position_variance ** 2)
    R_a = eye(2, dtype="float32") * (acceleration_variance ** 2)

    kalman = KalmanFilter(dim_x=6, dim_z=2)
    kalman.x = array([
        noised_positions[0, 0], 0, 0, noised_positions[0, 1], 0, 0,
    ], dtype="float32")
    # process noise
    kalman.Q = Q_discrete_white_noise(3, dt, process_variance, block_size=2)
    # initial uncertainty
    kalman.P *= acceleration_variance ** 2
    # state transition
    kalman.F = array([
        [1, dt, 0.5 * (dt ** 2), 0, 0, 0],
        [0, 1, dt, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, dt, 0.5 * (dt ** 2)],
        [0, 0, 0, 0, 1, dt],
        [0, 0, 0, 0, 0, 1],
    ], dtype="float32")

    gnss_i = 0
    for t in range(1, absolute_accelerations.shape[0]):
        kalman.predict()
        print(absolute_accelerations[t])
        kalman.update(absolute_accelerations[t], R=R_a, H=H_a)
        if (
            gnss_i < gnss_time.shape[0]
            and abs(acc_time[t] - gnss_time[gnss_i]) < 1e-4
        ):
            kalman.update(noised_positions[gnss_i], R=R_p, H=H_p)
            gnss_i += 1

        corrected_positions[t, 0] = kalman.x[0]
        corrected_positions[t, 1] = kalman.x[3]

    print(
        f"Positions: {positions.shape[0]}, "
        f"Corrected: {corrected_positions.shape[0]}"
    )
    pyplot.plot(positions[:, 0], positions[:, 1], label="Original")
    pyplot.plot(noised_positions[:, 0], noised_positions[:, 1], label="Noised")
    pyplot.plot(corrected_positions[:, 0], corrected_positions[:, 1], label="Fused")
    pyplot.legend()
    pyplot.show()


if __name__ == "__main__":
    localize()
