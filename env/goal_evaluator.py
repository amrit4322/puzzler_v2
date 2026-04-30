import math


class GoalEvaluator:
    def __init__(self, level_data, objects):
        self.level_data = level_data
        self.objects = objects
        self.goal_memory = {}

    def reset(self):
        self.goal_memory = {}

    def check_success(self):
        goals = self.level_data.get("goals", [])

        if not goals:
            return False

        mode = self.level_data.get("goal_logic", "all")

        if mode == "any":
            return any(self._check_goal(goal) for goal in goals)

        return all(self._check_goal(goal) for goal in goals)

    def _check_goal(self, goal):
        goal_type = goal["type"]

        if goal_type == "object_reaches_region":
            return self._object_reaches_region(goal)

        if goal_type == "object_stays_in_region":
            return self._object_stays_in_region(goal)

        if goal_type == "object_angle_range":
            return self._object_angle_range(goal)

        if goal_type == "object_above_height":
            return self._object_above_height(goal)

        if goal_type == "object_below_height":
            return self._object_below_height(goal)

        if goal_type == "object_velocity_below":
            return self._object_velocity_below(goal)

        raise ValueError(f"Unsupported goal type: {goal_type}")

    def _get_object_body(self, object_id):
        if object_id not in self.objects:
            raise KeyError(f"Object '{object_id}' not found in objects dictionary.")

        return self.objects[object_id]["body"]

    def _object_reaches_region(self, goal):
        object_id = goal["object_id"]
        region = goal["region"]

        obj = self.objects[object_id]
        body = obj["body"]

        x = float(body.position.x)
        y = float(body.position.y)

        object_radius = 0.0
        if obj["type"] == "circle":
            object_radius = float(obj["config"].get("radius", 0.0))

        if region["shape"] == "circle":
            cx = float(region["x"])
            cy = float(region["y"])
            r = float(region["r"])

            dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)

            # Uses overlap-style success:
            # object reaches goal when the object circle touches/enters the goal.
            return dist <= r + object_radius

        if region["shape"] == "rectangle":
            rx = float(region["x"])
            ry = float(region["y"])
            w = float(region["width"])
            h = float(region["height"])

            return rx <= x <= rx + w and ry <= y <= ry + h

        raise ValueError(f"Unsupported region shape: {region['shape']}")

    def _object_stays_in_region(self, goal):
        goal_id = goal["id"]
        duration_steps = int(goal.get("duration_steps", 60))

        inside = self._object_reaches_region(goal)

        if inside:
            self.goal_memory[goal_id] = self.goal_memory.get(goal_id, 0) + 1
        else:
            self.goal_memory[goal_id] = 0

        return self.goal_memory[goal_id] >= duration_steps

    def _object_angle_range(self, goal):
        object_id = goal["object_id"]
        min_angle = float(goal["min_angle"])
        max_angle = float(goal["max_angle"])

        body = self._get_object_body(object_id)
        angle = float(body.angle)

        return min_angle <= angle <= max_angle

    def _object_above_height(self, goal):
        object_id = goal["object_id"]
        y_threshold = float(goal["y"])

        body = self._get_object_body(object_id)

        # In pygame-style coordinates, smaller y means higher on screen.
        return float(body.position.y) <= y_threshold

    def _object_below_height(self, goal):
        object_id = goal["object_id"]
        y_threshold = float(goal["y"])

        body = self._get_object_body(object_id)

        # In pygame-style coordinates, larger y means lower on screen.
        return float(body.position.y) >= y_threshold

    def _object_velocity_below(self, goal):
        object_id = goal["object_id"]
        max_speed = float(goal.get("max_speed", 10.0))

        body = self._get_object_body(object_id)
        speed = body.velocity.length

        return speed <= max_speed