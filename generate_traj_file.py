import numpy as np
import traceback
import os

from utils import read_dfs


def translation_rotation_to_matrix(
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    degrees: bool = True,
) -> np.ndarray:
    """
    Convert translation and rotation parameters to a 4x4 transformation matrix.

    :param x: X coordinate for translation
    :type x: float
    :param y: Y coordinate for translation
    :type y: float
    :param z: Z coordinate for translation
    :type z: float
    :param roll: Rotation around X-axis
    :type roll: float
    :param pitch: Rotation around Y-axis
    :type pitch: float
    :param yaw: Rotation around Z-axis
    :type yaw: float
    :param degrees: Whether the rotation angles are in degrees (True) or radians (False)
    :type degrees: bool
    :return: 4x4 homogeneous transformation matrix
    :rtype: np.ndarray
    :raises ValueError: If input parameters are not valid numbers
    """
    try:
        if degrees:
            roll = np.deg2rad(roll, dtype=np.float64)
            pitch = np.deg2rad(pitch, dtype=np.float64)
            yaw = np.deg2rad(yaw, dtype=np.float64)

        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])

        R = Rz @ Ry @ Rx

        T = np.identity(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = [x, y, z]

        return T

    except (TypeError, ValueError) as e:
        traceback.print_exc()
        raise ValueError(f"Invalid input parameters for transformation matrix: {e}")


def main():
    DATA_FOLDER = os.path.join("D:", "Documentos", "Datasets", "Robot@VirtualHomeLarge")

    for home_number in range(1, 30):
        home_name = f"Home{home_number:02d}"
        for experiment in [
            # "Wandering",
            # "Wandering1",
            # "Wandering2",
            # "Wandering3",
            # "CustomWandering",
            # "CustomWandering2",
            "Wandering",
        ]:
            print(f"Processing {experiment} experiment - {home_name} ...")

            image_data_file = "LogImg.csv"

            if experiment == "Grid":
                image_data_file = "InfoGrid.csv"

            data_path = os.path.join(DATA_FOLDER, home_name, experiment)
            trajectory_path = os.path.join(data_path, "traj.txt")
            log_scan_df, filtered_info_grid_df, virtual_objs_df = read_dfs(
                image_data_file, data_path
            )

            traj_data = []
            for row in log_scan_df.itertuples():
                x, y, z = row.robot_position
                roll, pitch, yaw = row.robot_rotation

                # yaw_offset = 0
                # if "Custom" in experiment:
                #     yaw_offset = 180

                # x = x
                # y = -y
                # yaw = -yaw + yaw_offset

                c2w = translation_rotation_to_matrix(
                    x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, degrees=True
                )
                traj_data.append(" ".join([f"{el:.18e}" for el in c2w.flatten()]))
            with open(trajectory_path, "w") as f:
                f.write("\n".join(traj_data))


if __name__ == "__main__":
    main()
