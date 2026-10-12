"""Smooth forward sensor likelihoods in world coordinates."""
import numpy as np


def detection_probabilities(points, positions, headings, active, sensor_range,
                            fov_degrees, peak_probability):
    """Per-drone, per-point probabilities for one sensing opportunity.

    Raised cosines go smoothly to zero at the range and angular limits.
    The exact same field is used for real detections and belief integration.
    A 360-degree sensor has no angular attenuation.
    """
    delta = np.asarray(points)[None] - np.asarray(positions)[:, None]
    distance = np.linalg.norm(delta, axis=-1)
    bearing = np.arctan2(delta[..., 1], delta[..., 0])-np.asarray(headings)[:, None]
    angle = np.abs(np.arctan2(np.sin(bearing), np.cos(bearing)))
    radial = np.cos(np.pi/2*np.minimum(distance/sensor_range, 1.))**2
    radial = np.where(distance < sensor_range, radial, 0.)
    if fov_degrees == 360:
        angular = np.ones_like(radial)
    else:
        half_angle = np.deg2rad(fov_degrees)/2
        angular = np.cos(np.pi/2*np.minimum(angle/half_angle, 1.))**2
        angular = np.where(angle < half_angle, angular, 0.)
        angular = np.where(distance < 1e-12, 1., angular)
    return peak_probability*radial*angular*np.asarray(active)[:, None]


def cell_detection_probabilities(centers, width, positions, headings, active,
                                 sensor_range, fov_degrees, peak_probability):
    """Approximate union detection probability averaged over each square cell.

    Nine Gauss-Legendre samples integrate partial sensor coverage. Combine
    the drones at each sample before integrating, including overlapping FoVs.
    """
    nodes = np.array([-np.sqrt(3/5), 0., np.sqrt(3/5)])*width/2
    weights = np.array([5/18, 4/9, 5/18])
    offsets = np.stack(np.meshgrid(nodes, nodes), -1).reshape(-1, 2)
    sample_weights = np.outer(weights, weights).ravel()
    samples = (np.asarray(centers)[:, None]+offsets).reshape(-1, 2)
    pd = detection_probabilities(samples, positions, headings, active,
                                 sensor_range, fov_degrees, peak_probability)
    combined = 1-np.prod(1-pd, axis=0)
    return combined.reshape(-1, 9)@sample_weights
