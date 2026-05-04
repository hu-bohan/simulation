import math

import numpy as np


class NavigationObstacleTerrain:
    """Triangle-mesh terrain with static navigation obstacles.

    The mesh is intentionally not implemented as Isaac actors. It exposes the
    same attributes used by LeggedRobot._create_trimesh, while also storing the
    local obstacle layout that the navigation environment can copy into its
    virtual obstacle buffers.
    """

    def __init__(self, terrain_cfg, navigation_cfg, num_robots):
        self.cfg = terrain_cfg
        self.nav_cfg = navigation_cfg
        self.num_robots = num_robots
        self.type = "trimesh"

        self.num_rows = int(max(1, terrain_cfg.num_rows))
        self.num_cols = int(max(1, terrain_cfg.num_cols))
        self.cfg.num_sub_terrains = self.num_rows * self.num_cols
        self.border_size = float(terrain_cfg.border_size)
        self.horizontal_scale = float(terrain_cfg.horizontal_scale)

        self.start_margin = float(getattr(navigation_cfg, "terrain_nav_start_margin", 2.0))
        self.end_margin = float(getattr(navigation_cfg, "terrain_nav_end_margin", 2.0))
        self.side_margin = float(getattr(navigation_cfg, "terrain_nav_side_margin", 2.0))

        min_length = float(navigation_cfg.field_length) + self.start_margin + self.end_margin
        min_width = float(navigation_cfg.field_width) + 2.0 * self.side_margin
        self.env_length = max(float(terrain_cfg.terrain_length), min_length)
        self.env_width = max(float(terrain_cfg.terrain_width), min_width)
        self.length_per_env_pixels = int(math.ceil(self.env_length / self.horizontal_scale))
        self.width_per_env_pixels = int(math.ceil(self.env_width / self.horizontal_scale))
        self.border = int(math.ceil(self.border_size / self.horizontal_scale))

        self.origin_x_in_tile = self.start_margin
        self.origin_y_in_tile = 0.5 * self.env_width
        self.env_origins = self._build_env_origins()

        self.nav_obstacle_positions = np.zeros(
            (self.num_rows, self.num_cols, navigation_cfg.num_obstacles, 2),
            dtype=np.float32,
        )
        self.nav_obstacle_radii = np.zeros(
            (self.num_rows, self.num_cols, navigation_cfg.num_obstacles),
            dtype=np.float32,
        )
        self.obstacle_seed = self._resolve_obstacle_seed()
        self._build_obstacle_layouts()
        self.vertices, self.triangles = self._build_mesh()

        self.heightsamples = self._build_flat_height_samples()
        self.height_field_raw = self.heightsamples
        self.tot_rows, self.tot_cols = self.heightsamples.shape

    def get_obstacles(self, row, col):
        row = int(np.clip(row, 0, self.num_rows - 1))
        col = int(np.clip(col, 0, self.num_cols - 1))
        return (
            self.nav_obstacle_positions[row, col].copy(),
            self.nav_obstacle_radii[row, col].copy(),
        )

    def _build_env_origins(self):
        origins = np.zeros((self.num_rows, self.num_cols, 3), dtype=np.float32)
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                origins[row, col, 0] = row * self.env_length + self.origin_x_in_tile
                origins[row, col, 1] = col * self.env_width + self.origin_y_in_tile
        return origins

    def _build_obstacle_layouts(self):
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                rng = np.random.default_rng(self.obstacle_seed + row * self.num_cols + col)
                positions, radii = self._sample_obstacles(rng)
                self.nav_obstacle_positions[row, col] = positions
                self.nav_obstacle_radii[row, col] = radii

    def _resolve_obstacle_seed(self):
        seed = getattr(self.nav_cfg, "terrain_obstacle_seed", None)
        if seed is None or int(seed) < 0:
            return int(np.random.default_rng().integers(0, np.iinfo(np.int32).max))
        return int(seed)

    def _sample_obstacles(self, rng):
        fixed_layout = self._fixed_obstacle_layout()
        if fixed_layout is not None:
            return fixed_layout

        nav_cfg = self.nav_cfg
        obstacles = []
        start_xy = np.array([0.0, nav_cfg.path_center_y], dtype=np.float32)
        goal_xy = np.array([nav_cfg.field_length, nav_cfg.path_center_y], dtype=np.float32)

        if getattr(nav_cfg, "path_type", "line") == "sine":
            goal_xy[1] = self._path_y_at_x(nav_cfg.field_length)

        for _ in range(nav_cfg.num_obstacles):
            path_biased = rng.random() < nav_cfg.obstacle_path_bias
            for _attempt in range(200):
                radius = float(rng.uniform(nav_cfg.obstacle_radius_range[0], nav_cfg.obstacle_radius_range[1]))
                x_pos = float(rng.uniform(nav_cfg.obstacle_x_range[0], nav_cfg.obstacle_x_range[1]))

                if path_biased:
                    y_pos = self._path_y_at_x(x_pos) + float(
                        rng.uniform(-nav_cfg.obstacle_path_offset, nav_cfg.obstacle_path_offset)
                    )
                else:
                    y_pos = float(
                        rng.uniform(
                            nav_cfg.obstacle_margin + radius,
                            nav_cfg.field_width - nav_cfg.obstacle_margin - radius,
                        )
                    )

                candidate = np.array([x_pos, y_pos], dtype=np.float32)
                if np.linalg.norm(candidate - start_xy) < 2.0 + radius:
                    continue
                if np.linalg.norm(candidate - goal_xy) < 1.8 + radius:
                    continue
                if y_pos < nav_cfg.obstacle_margin + radius:
                    continue
                if y_pos > nav_cfg.field_width - nav_cfg.obstacle_margin - radius:
                    continue

                valid = True
                for existing_pos, existing_radius in obstacles:
                    spacing = radius + existing_radius + nav_cfg.obstacle_min_spacing
                    if np.linalg.norm(candidate - existing_pos) < spacing:
                        valid = False
                        break
                if valid:
                    obstacles.append((candidate, radius))
                    break

        while len(obstacles) < nav_cfg.num_obstacles:
            obstacles.append((np.array([-100.0, -100.0], dtype=np.float32), 0.0))

        positions = np.stack([item[0] for item in obstacles], axis=0).astype(np.float32)
        radii = np.array([item[1] for item in obstacles], dtype=np.float32)
        return positions, radii

    def _fixed_obstacle_layout(self):
        layout = getattr(self.nav_cfg, "terrain_obstacle_layout", None)
        if layout is None:
            return None

        entries = np.asarray(layout, dtype=np.float32)
        if entries.size == 0:
            entries = np.zeros((0, 3), dtype=np.float32)
        elif entries.ndim == 1:
            entries = entries.reshape(1, -1)

        if entries.ndim != 2 or entries.shape[1] != 3:
            raise ValueError("navigation.terrain_obstacle_layout must contain [x, y, radius] rows")

        positions = np.full((self.nav_cfg.num_obstacles, 2), -100.0, dtype=np.float32)
        radii = np.zeros((self.nav_cfg.num_obstacles,), dtype=np.float32)

        count = min(self.nav_cfg.num_obstacles, entries.shape[0])
        for idx in range(count):
            x_pos, y_pos, radius = entries[idx]
            if radius <= 0.0:
                continue
            positions[idx] = [x_pos, y_pos]
            radii[idx] = radius
        return positions, radii

    def _path_y_at_x(self, x_value):
        nav_cfg = self.nav_cfg
        x_value = float(np.clip(x_value, 0.0, nav_cfg.field_length))
        if getattr(nav_cfg, "path_type", "line") == "sine":
            return nav_cfg.path_center_y + nav_cfg.path_amplitude * math.sin(
                2.0 * math.pi * x_value / nav_cfg.path_wavelength
            )
        return nav_cfg.path_center_y

    def _build_mesh(self):
        vertices = []
        triangles = []
        self._append_ground(vertices, triangles)

        obstacle_height = float(getattr(self.nav_cfg, "terrain_obstacle_height", 0.8))
        segments = int(max(8, getattr(self.nav_cfg, "terrain_obstacle_segments", 24)))
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                origin = self.env_origins[row, col]
                for obs_pos, radius in zip(
                    self.nav_obstacle_positions[row, col],
                    self.nav_obstacle_radii[row, col],
                ):
                    if radius <= 0.0:
                        continue
                    center_x = origin[0] + obs_pos[0]
                    center_y = origin[1] + obs_pos[1] - 0.5 * self.nav_cfg.field_width
                    self._append_cylinder(vertices, triangles, center_x, center_y, radius, obstacle_height, segments)

        return np.asarray(vertices, dtype=np.float32), np.asarray(triangles, dtype=np.uint32)

    def _append_ground(self, vertices, triangles):
        min_x = 0.0
        min_y = 0.0
        max_x = self.num_rows * self.env_length
        max_y = self.num_cols * self.env_width
        base = len(vertices)
        vertices.extend(
            [
                [min_x + self.border_size, min_y + self.border_size, 0.0],
                [max_x + self.border_size, min_y + self.border_size, 0.0],
                [max_x + self.border_size, max_y + self.border_size, 0.0],
                [min_x + self.border_size, max_y + self.border_size, 0.0],
            ]
        )
        triangles.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])

    def _append_cylinder(self, vertices, triangles, center_x, center_y, radius, height, segments):
        center_x += self.border_size
        center_y += self.border_size
        base = len(vertices)
        for idx in range(segments):
            angle = 2.0 * math.pi * idx / segments
            x_pos = center_x + radius * math.cos(angle)
            y_pos = center_y + radius * math.sin(angle)
            vertices.append([x_pos, y_pos, 0.0])
            vertices.append([x_pos, y_pos, height])

        top_center = len(vertices)
        vertices.append([center_x, center_y, height])

        for idx in range(segments):
            next_idx = (idx + 1) % segments
            bottom = base + 2 * idx
            top = bottom + 1
            next_bottom = base + 2 * next_idx
            next_top = next_bottom + 1
            triangles.append([bottom, next_bottom, top])
            triangles.append([top, next_bottom, next_top])
            triangles.append([top_center, top, next_top])

    def _build_flat_height_samples(self):
        rows = int(math.ceil(self.num_rows * self.env_length / self.horizontal_scale))
        cols = int(math.ceil(self.num_cols * self.env_width / self.horizontal_scale))
        return np.zeros((rows + 2 * self.border, cols + 2 * self.border), dtype=np.int16)
