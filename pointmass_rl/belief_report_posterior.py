"""Public Gaussian report belief with uncertain centre, scale and association.

The simulator's realized centre/scale never enter this filter. We maintain a
joint posterior over Gaussian scale, formation centre and remaining type
counts. Members share that latent centre, so finding one localizes the rest.
Spatial integration uses histogram cells; inter-member exclusion distances
are not represented. Unreported detections have a nuisance likelihood, but do
not create spatial search mass before they are observed.
"""
from math import erfc

import numpy as np


def normal_interval(low, high):
    """Standard Gaussian mass in [low, high], including small tail intervals."""
    complementary = np.vectorize(erfc, otypes=[float])
    # Symmetry avoids subtracting two numbers rounded to 1 in positive tails.
    positive = low >= 0
    a = np.where(positive, low, -high)/np.sqrt(2)
    b = np.where(positive, high, -low)/np.sqrt(2)
    return np.maximum(0., .5*(complementary(a)-complementary(b)))


class ReportPosterior:
    def __init__(self, config, points, report):
        self.c = config
        self.points = np.asarray(points)*config.size
        self.shape = points.shape[:-1]
        self.centers = self.points.reshape(-1,2)
        self.width = (config.size-2*config.belief_search_margin)/(
            config.belief_grid*config.belief_subcells)
        counts = tuple(int(x) for x in report['type_counts'])
        if len(counts) != 3 or min(counts) < 0 or not 1 <= sum(counts) <= config.n_targets:
            raise ValueError('Initial report requires valid type counts')

        # Quadrature over a broad public prior on log(sigma). The actual
        # episode sigma is neither a quadrature node nor supplied to us.
        hint, spread = float(report['sigma_hint']), float(report['log_sigma_std'])
        if not np.isfinite(hint) or hint <= 0 or not np.isfinite(spread) or not 0 <= spread <= 1.5:
            raise ValueError('Invalid Gaussian scale prior in initial report')
        nodes, weights = np.polynomial.hermite.hermgauss(7 if spread else 1)
        self.scales = hint*np.exp(np.sqrt(2)*spread*nodes)
        self.scale_prior = weights/np.sqrt(np.pi)
        delta = self.centers-np.asarray(report['position'],dtype=float)
        sigma = self.scales[:,None,None]
        center_mass = normal_interval((delta-self.width/2)/sigma,
                                      (delta+self.width/2)/sigma).prod(-1)
        center_mass /= center_mass.sum(-1,keepdims=True)
        # Truncation is conditional on each scale: map boundaries do not
        # silently change the public prior probability of a sigma hypothesis.
        self.hypotheses = {counts:self.scale_prior[:,None]*center_mass}
        self._member_support(float(report['member_radius']))

        totals = (np.arange(config.min_targets,config.n_targets+1)
                  if config.randomize_counts else np.array([config.n_targets]))
        possible = totals[(totals+1)//2 == sum(counts)]
        if not len(possible):
            raise ValueError('Report count is inconsistent with the scenario configuration')
        expected = float(np.mean(possible-sum(counts)))
        # The unreported formation is encountered incidentally. Its expected
        # count only supplies an alternative source for a new detection; it
        # does not contribute search actions or spatial belief mass.
        area = (config.size-2*config.belief_search_margin)**2
        self.unreported_likelihood = expected/(3*area)
        self._refresh()

    def _member_support(self, radius):
        """Sparse conditional member histograms for each possible centre.

        Both the centre and the member represent a finite cell. The boundary
        ramp integrates their unresolved offsets approximately and includes
        every cell that can contain a member of a valid continuous formation.
        """
        padding = np.sqrt(2)*self.width
        side = self.c.belief_grid*self.c.belief_subcells
        margin = self.c.belief_search_margin
        coordinates = np.rint((self.centers-margin)/self.width-.5).astype(int)
        lookup = np.empty((side,side),dtype=int)
        lookup[coordinates[:,1],coordinates[:,0]] = np.arange(len(self.centers))
        span = int(np.ceil((radius+padding)/self.width))
        axis = np.arange(-span,span+1)
        offsets = np.stack(np.meshgrid(axis,axis),-1).reshape(-1,2)
        distances = np.linalg.norm(offsets*self.width,axis=-1)
        keep = distances < radius+padding
        offsets,distances = offsets[keep],distances[keep]
        neighbours = coordinates[:,None]+offsets
        valid = ((neighbours >= 0) & (neighbours < side)).all(-1)
        safe = np.clip(neighbours,0,side-1)
        self.member_indices = lookup[safe[...,1],safe[...,0]]
        density = np.clip((radius+padding-distances)/(2*padding),0,1)
        self.members = valid*density[None]
        self.members /= self.members.sum(-1,keepdims=True)

    def _measurement_likelihood(self, position):
        delta = np.asarray(position)-self.points
        sigma = self.c.position_noise
        if sigma <= 1e-12:
            inside = ((delta >= -self.width/2) & (delta < self.width/2)).all(-1)
            return inside.astype(float)/(self.width*self.width)
        return normal_interval((delta-self.width/2)/sigma,
                               (delta+self.width/2)/sigma).prod(-1)/(self.width*self.width)

    def update(self, detection_map, new_observations):
        # Sum over report/background association jointly with centre and scale.
        # Use predictive densities before clearing the remaining search area.
        for _,position,typ,_ in new_observations:
            index = int(typ)-1
            likelihood = self._measurement_likelihood(position)
            reported = (self.members*likelihood.ravel()[self.member_indices]).sum(-1)
            unreported = self.unreported_likelihood
            updated = {}
            for counts,weight in self.hypotheses.items():
                updated[counts] = updated.get(counts,0.)+weight*unreported
                if counts[index]:
                    remaining = list(counts)
                    remaining[index] -= 1
                    remaining = tuple(remaining)
                    updated[remaining] = (updated.get(remaining,0.)
                                          +weight*counts[index]*reported[None])
            total = sum(weight.sum() for weight in updated.values())
            if total <= 0:
                raise RuntimeError('New detection has zero likelihood under the public search belief')
            self.hypotheses = {counts:weight/total for counts,weight in updated.items()
                               if weight.any()}

        miss = 1-np.clip(detection_map,0,1)
        members = self.members*miss.ravel()[self.member_indices]
        survival = members.sum(-1)
        # Evidence for every remaining member, conditional on a shared centre.
        # Empty observations update sigma as well as position.
        with np.errstate(divide='ignore'):
            log_survival = np.log(survival)
            logs = {counts:np.log(weight)+(sum(counts)*log_survival[None] if sum(counts) else 0.)
                    for counts,weight in self.hypotheses.items()}
        largest = max(values.max() for values in logs.values())
        if not np.isfinite(largest):
            raise RuntimeError('Perfect coverage contradicts the known remaining report count')
        weights = {counts:np.exp(values-largest) for counts,values in logs.items()}
        total = sum(weight.sum() for weight in weights.values())
        self.hypotheses = {counts:weight/total for counts,weight in weights.items() if weight.any()}
        self.members = np.divide(members,survival[:,None],out=np.zeros_like(members),
                                 where=survival[:,None] > 0)
        self._refresh()

    def _refresh(self):
        joint = sum(self.hypotheses.values())
        self.scale_posterior = joint.sum(-1)
        self.center_posterior = joint.sum(0)
        self.sigma_mean = float(self.scale_posterior@self.scales)
        self.sigma_std = float(np.sqrt(self.scale_posterior@(self.scales-self.sigma_mean)**2))
        self.center_mean = self.center_posterior@self.centers
        delta = self.centers-self.center_mean
        self.center_covariance = (delta*self.center_posterior[:,None]).T@delta
        type_counts = sum((weight.sum(0)[:,None]*np.asarray(counts)
                           for counts,weight in self.hypotheses.items()),
                          np.zeros((len(self.centers),3)))
        self.remaining_types = type_counts.sum(0)
        reported = np.stack([
            np.bincount(self.member_indices.ravel(),
                        weights=(self.members*type_counts[:,t,None]).ravel(),
                        minlength=len(self.centers)) for t in range(3)],axis=-1).reshape((*self.shape,3))
        remaining = float(self.remaining_types.sum())
        self.spatial = reported.sum(-1)/remaining if remaining > 0 else np.zeros(self.shape)
        self.type_weights = reported
