from typing import Tuple, Any
import numpy as np
import threading
import cv2


class MapNavigator(threading.Thread):
    """
    Interactive map navigator thread for visualizing and controlling user position.
    """

    def __init__(
        self,
        window_name: str,
        base_image: np.ndarray,
        origin: Tuple[float, float],
        resolution: float,
        initial_pose: Tuple[float, float, float],
    ) -> None:
        """
        Initializes the map navigator.

        :param window_name: Name of the display window.
        :type window_name: str
        :param base_image: Base map image.
        :type base_image: np.ndarray
        :param origin: Map origin coordinates (x, y).
        :type origin: Tuple[float, float]
        :param resolution: Map resolution in meters per pixel.
        :type resolution: float
        :param initial_pose: Initial user world coordinates (x, y, z).
        :type initial_pose: Tuple[float, float, float]
        """
        super().__init__(daemon=True)
        self.window_name = window_name
        self.base_image = base_image
        self.origin = origin
        self.resolution = resolution
        self.height, self.width = base_image.shape[:2]

        self._user_pos = initial_pose
        self._lock = threading.Lock()
        self._running = True
        self._should_exit = False
        self._change_room_callback = None

    def set_change_room_callback(self, callback: Any) -> None:
        """
        Sets the callback function to be called when changing rooms.

        :param callback: Function to call on room change.
        :type callback: Any
        """
        self._change_room_callback = callback

    def _change_room_signal(self) -> None:
        """
        Changes the current room and invokes the callback if set.
        """
        if self._change_room_callback:
            self._change_room_callback(self._user_pos)

    def _world_to_map_coordinates(
        self, world_coords: Tuple[float, float, float]
    ) -> Tuple[int, int]:
        """
        Internal conversion from world to map coordinates.

        :param world_coords: World coordinates (x, y, z).
        :type world_coords: Tuple[float, float, float]
        :return: Pixel coordinates (x, y).
        :rtype: Tuple[int, int]
        """
        world_x_map = world_coords[2]
        world_y_map = -world_coords[0]
        pixel_x = int((world_x_map - self.origin[0]) / self.resolution)
        pixel_y = int(self.height - ((world_y_map - self.origin[1]) / self.resolution))
        return pixel_x, pixel_y

    def _map_to_world_coordinates(
        self, pixel_coords: Tuple[int, int]
    ) -> Tuple[float, float, float]:
        """
        Internal conversion from map to world coordinates.

        :param pixel_coords: Pixel coordinates (x, y).
        :type pixel_coords: Tuple[int, int]
        :return: World coordinates (x, y, z).
        :rtype: Tuple[float, float, float]
        """
        px, py = pixel_coords
        world_y_map = ((self.height - py) * self.resolution) + self.origin[1]
        world_x_map = (px * self.resolution) + self.origin[0]

        raw_x = -world_y_map
        raw_y = 0.0
        raw_z = world_x_map
        return (raw_x, raw_y, raw_z)

    def _draw_user_on_map(
        self, image: np.ndarray, user_pos: Tuple[float, float, float]
    ) -> np.ndarray:
        """
        Draws user position on the map image.

        :param image: Base map image.
        :type image: np.ndarray
        :param user_pos: User world coordinates (x, y, z).
        :type user_pos: Tuple[float, float, float]
        :return: Map image with user marker.
        :rtype: np.ndarray
        """
        img_copy = image.copy()
        px, py = self._world_to_map_coordinates(user_pos)

        if 0 <= px < self.width and 0 <= py < self.height:
            cv2.circle(img_copy, (px, py), 6, (255, 0, 0), -1)
            cv2.circle(img_copy, (px, py), 4, (0, 0, 255), -1)
            cv2.putText(
                img_copy,
                "USER",
                (px + 8, py),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                3,
            )
            cv2.putText(
                img_copy,
                "USER",
                (px + 8, py),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
            )
        return img_copy

    def run(self) -> None:
        """
        Main thread loop for rendering the map and handling input.

        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while self._running:
            with self._lock:
                current_pos = self._user_pos

            img_to_show = self._draw_user_on_map(self.base_image, current_pos)
            cv2.imshow(self.window_name, img_to_show)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                self._should_exit = True
                self._running = False

        cv2.destroyWindow(self.window_name)

    def mouse_callback(
        self, event: int, x: int, y: int, flags: int, param: Any
    ) -> None:
        """
        Handles mouse click events to move the user position.

        :param event: OpenCV mouse event type.
        :type event: int
        :param x: Mouse x coordinate.
        :type x: int
        :param y: Mouse y coordinate.
        :type y: int
        :param flags: Additional mouse event flags.
        :type flags: int
        :param param: Additional callback parameters.
        :type param: Any
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if 0 <= y < self.height and 0 <= x < self.width:
                pixel_color = self.base_image[y, x]
                if np.all(pixel_color == 0):
                    return  # Invalid area

                new_pos = self._map_to_world_coordinates((x, y))
                with self._lock:
                    self._user_pos = new_pos
                    self._change_room_signal()

    def move_to_coordinate(self, world_coords: Tuple[float, float, float]) -> None:
        """
        Moves user to specified world coordinates with validation.

        :param world_coords: Target world coordinates (x, y, z).
        :type world_coords: Tuple[float, float, float]
        """
        px, py = self._world_to_map_coordinates(world_coords)

        # Check if coordinates are within bounds
        if not (0 <= px < self.width and 0 <= py < self.height):
            return

        target_color = self.base_image[py, px]

        # Check if target is valid (not black)
        if not np.all(target_color == 0):
            with self._lock:
                self._user_pos = world_coords
            return

        # Spiral search for nearest valid point
        max_radius = 50
        found_valid = False
        best_px, best_py = px, py

        for r in range(1, max_radius + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if abs(dx) != r and abs(dy) != r:
                        continue

                    nx, ny = px + dx, py + dy

                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        if not np.all(self.base_image[ny, nx] == 0):
                            best_px, best_py = nx, ny
                            found_valid = True
                            break
                if found_valid:
                    break
            if found_valid:
                break

        if found_valid:
            corrected_world_pos = self._map_to_world_coordinates((best_px, best_py))
            with self._lock:
                self._user_pos = corrected_world_pos

    @property
    def user_pos(self) -> Tuple[float, float, float]:
        """
        Gets current user position.

        :return: User world coordinates (x, y, z).
        :rtype: Tuple[float, float, float]
        """
        with self._lock:
            return self._user_pos

    @user_pos.setter
    def user_pos(self, val: Tuple[float, float, float]) -> None:
        """
        Sets user position.

        :param val: New user world coordinates (x, y, z).
        :type val: Tuple[float, float, float]
        """
        with self._lock:
            self._user_pos = val

    def should_exit(self) -> bool:
        """
        Checks if the navigator should exit.

        :return: True if exit requested, False otherwise.
        :rtype: bool
        """
        return self._should_exit

    def stop(self) -> None:
        """
        Stops the navigator thread.

        """
        self._running = False
