"""Forward-only circular-arc/straight paths for the game movement controller."""
import math

import numpy as np


TAU = 2 * math.pi


def advance(position, heading, distance, turn, radius):
    """Integrate one constant-curvature segment exactly."""
    if turn == 0:
        return position + distance * np.array([math.cos(heading), math.sin(heading)]), heading
    curvature = turn / radius
    end = heading + curvature * distance
    position = position + np.array([math.sin(end)-math.sin(heading),
                                    math.cos(heading)-math.cos(end)]) / curvature
    return position, math.atan2(math.sin(end), math.cos(end))


class DubinsPath:
    """Choose the shortest of the six Dubins families for an oriented goal."""
    def __init__(self, position, heading, goal, end_heading, radius):
        delta = np.asarray(goal) - position
        d = float(np.linalg.norm(delta)) / radius
        bearing = math.atan2(delta[1], delta[0])
        a, b = (heading-bearing) % TAU, (end_heading-bearing) % TAU
        sa, sb, ca, cb = math.sin(a), math.sin(b), math.cos(a), math.cos(b)
        cab = math.cos(a-b)
        candidates = []

        p2 = 2+d*d-2*cab+2*d*(sa-sb)
        if p2 >= -1e-10:
            angle = math.atan2(cb-ca, d+sa-sb)
            candidates.append(((1, 0, 1), ((angle-a)%TAU, math.sqrt(max(0,p2)), (b-angle)%TAU)))
        p2 = 2+d*d-2*cab+2*d*(sb-sa)
        if p2 >= -1e-10:
            angle = math.atan2(ca-cb, d-sa+sb)
            candidates.append(((-1, 0, -1), ((a-angle)%TAU, math.sqrt(max(0,p2)), (angle-b)%TAU)))
        p2 = d*d-2+2*cab+2*d*(sa+sb)
        if p2 >= -1e-10:
            p = math.sqrt(max(0,p2))
            angle = math.atan2(-ca-cb, d+sa+sb)-math.atan2(-2,p)
            candidates.append(((1, 0, -1), ((angle-a)%TAU, p, (angle-b)%TAU)))
        p2 = d*d-2+2*cab-2*d*(sa+sb)
        if p2 >= -1e-10:
            p = math.sqrt(max(0,p2))
            angle = math.atan2(ca+cb, d-sa-sb)-math.atan2(2,p)
            candidates.append(((-1, 0, 1), ((a-angle)%TAU, p, (b-angle)%TAU)))
        cosine = (6-d*d+2*cab+2*d*(sa-sb))/8
        if abs(cosine) <= 1+1e-10:
            middle = (TAU-math.acos(max(-1., min(1., cosine))))%TAU
            first = (a-math.atan2(ca-cb, d-sa+sb)+middle/2)%TAU
            candidates.append(((-1, 1, -1), (first, middle, (a-b-first+middle)%TAU)))
        cosine = (6-d*d+2*cab+2*d*(sb-sa))/8
        if abs(cosine) <= 1+1e-10:
            middle = (TAU-math.acos(max(-1., min(1., cosine))))%TAU
            first = (-a-math.atan2(ca-cb, d+sa-sb)+middle/2)%TAU
            candidates.append(((1, -1, 1), (first, middle, (b-a-first+middle)%TAU)))
        # A roundoff-sized negative angle must not become a full revolution.
        candidates = [(turns, tuple(0. if turn and min(length, TAU-length) < 1e-10 else length
                                    for turn, length in zip(turns, lengths)))
                      for turns, lengths in candidates]
        turns, lengths = min(candidates, key=lambda item: sum(item[1]))
        self.segments = [[turn, length*radius] for turn, length in zip(turns, lengths)]
        self.radius = radius
        self.length = sum(length for _, length in self.segments)

    @property
    def done(self):
        return not self.segments

    def move(self, position, heading, budget):
        while self.segments:
            turn, remaining = self.segments[0]
            distance = min(budget, remaining)
            position, heading = advance(position, heading, distance, turn, self.radius)
            budget -= distance
            remaining -= distance
            if remaining <= 1e-10:
                self.segments.pop(0)
            else:
                self.segments[0][1] = remaining
                break
        return position, heading, budget


def point_path(position, heading, goal, radius):
    delta = np.asarray(goal)-position
    bearing = math.atan2(delta[1], delta[0])
    angles = [heading, bearing, 0, math.pi/2, math.pi, -math.pi/2]
    # A point destination does not require a compass-aligned arrival. Include
    # the natural headings of tangent approaches from both initial circles.
    normal = np.array([-math.sin(heading), math.cos(heading)])
    for turn in (-1, 1):
        offset = np.asarray(goal)-(position+turn*radius*normal)
        distance = float(np.linalg.norm(offset))
        if distance >= radius:
            angles.append(math.atan2(offset[1], offset[0])+turn*math.asin(min(1.,radius/distance)))
    paths = [DubinsPath(position, heading, goal, angle, radius) for angle in angles]
    return min(paths, key=lambda path: path.length)


def lawnmower_waypoints(cell, config, position):
    """Overlapping horizontal sensor passes with oriented lane endpoints.

    Turns may leave the selected cell. Sensor evidence, rather than finishing
    the waypoint list, determines whether the region has been cleared.
    """
    margin = config.belief_search_margin
    width = (config.size-2*margin)/config.belief_grid
    x0 = margin+(cell % config.belief_grid)*width
    y0 = margin+(cell // config.belief_grid)*width
    half_angle = min(math.radians(config.sensor_fov_deg)/2, math.pi/2)
    spacing = min(width/config.belief_subcells,
                  config.sensor_range*math.sin(half_angle))
    rows = max(2, math.ceil(width/spacing))
    ys = y0+(np.arange(rows)+.5)*width/rows
    if abs(position[1]-ys[-1]) < abs(position[1]-ys[0]):
        ys = ys[::-1]
    # Extend lanes so a forward sensor also sees the near edge before entry.
    extension = config.sensor_range
    left, right = x0-extension, x0+width+extension
    east = abs(position[0]-left) <= abs(position[0]-right)
    result = []
    for y in ys:
        start, end, angle = (left, right, 0.) if east else (right, left, math.pi)
        result.extend([(np.array([start,y]), angle), (np.array([end,y]), angle)])
        east = not east
    return result
