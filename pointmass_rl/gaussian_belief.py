"""Continuous reported-object beliefs and an initially uniform background.

Only Gaussian parameters and discovered tracks are filter state. Integration
nodes are disposable numerical quadrature, never occupancy/subcell memory.
The negative-observation update projects changes in domain-conditioned moments
onto Gaussian parameters; it is approximate. Report-association uncertainty is
stored as probabilities over remaining-component subsets.
"""
import numpy as np


GAUSSIAN_FEATURES = 18


class LocalBackground:
    """Smooth overlapping Gaussian basis, with conservative local transport.

    The basis is a partition of unity: initial density is exactly uniform.
    Each continuous basis function is normalized over the search domain.
    Observation removes a fraction of its mass and transfers it to nearby,
    less-observed basis functions. There are no piecewise-constant cells.
    """
    def __init__(self, config, points, measure, display):
        self.c = config
        lo, hi = config.belief_search_margin, config.size-config.belief_search_margin
        axis = np.linspace(lo, hi, 9)
        self.centers = np.stack(np.meshgrid(axis, axis), -1).reshape(-1, 2)
        self.sigma = (hi-lo)/8
        raw = self.partition(points.reshape(-1, 2))
        self.integrals = measure.ravel()@raw
        self.basis = raw/self.integrals
        self.display_basis = self.partition(display.reshape(-1, 2))/self.integrals
        self.mass = self.integrals/(hi-lo)**2
        delta = self.centers[:, None]-self.centers[None]
        distance = np.square(delta).sum(-1)
        self.neighbours = np.exp(-distance/(2*(1.4*self.sigma)**2))
        self.neighbours[distance > (3*self.sigma)**2] = 0
        np.fill_diagonal(self.neighbours, 0)
        self.measure = measure.ravel()

    def partition(self, points):
        delta = points[:, None]-self.centers[None]
        values = np.exp(-np.square(delta).sum(-1)/(2*self.sigma**2))
        return values/np.maximum(values.sum(-1, keepdims=True), 1e-300)

    def update(self, probability):
        exposure = (self.measure*probability.ravel())@self.basis
        exposure = np.clip(exposure, 0, 1)
        removed = self.mass*exposure
        redistribution = self.neighbours*(1-exposure[None])
        denominator = redistribution.sum(-1, keepdims=True)
        redistribution = np.divide(redistribution, denominator,
                                   out=np.zeros_like(redistribution), where=denominator > 0)
        trapped = denominator[:, 0] <= 0
        removed[trapped] = 0
        self.mass = self.mass-removed+removed@redistribution
        self.mass /= self.mass.sum()

    def density(self, points):
        return (self.partition(points)/self.integrals)@self.mass


class GaussianTargetBelief:
    """Action records preserve the legacy first 14 field positions.

    Extra fields: uniform mass, region width/height (map-normalized), reported
    mass fraction. Type probabilities and expected life combine reported mass
    with the background attribute prior. Tracks have no uniform mass.
    """
    def __init__(self, config, initial_intel=None):
        self.c = config
        self.initial_intel = initial_intel
        self.cells = config.belief_grid**2
        self.capacity = self.cells + config.n_targets
        self.report = None  # Legacy histogram filter is not used.
        self.tracks = []
        self.components = []
        if initial_intel:
            counts = initial_intel['type_counts']
            if (len(counts) != 3 or any(int(x) != x or x < 0 for x in counts)
                    or not 1 <= sum(counts) <= config.n_targets):
                raise ValueError('Initial report requires valid type counts')
            # E[sigma**2] of the public log-normal scale prior, plus the
            # marginal variance of a member uniformly spread in a disk.
            variance = (initial_intel['sigma_hint']**2
                        * np.exp(2*initial_intel['log_sigma_std']**2)
                        + initial_intel['member_radius']**2/4)
            variance = max(variance, config.belief_gaussian_min_sigma**2)
            for typ, count in enumerate(counts, 1):
                for _ in range(int(count)):
                    self.components.append(dict(
                        mean=np.asarray(initial_intel['position'], dtype=float).copy(),
                        covariance=np.eye(2)*variance, type=typ, remaining=1.))
        self.background_mass = config.belief_background_mass
        self.background_type_prior = np.asarray(config.belief_background_type_prior, dtype=float)
        self.background_type_prior = self.background_type_prior/self.background_type_prior.sum()
        # Hypotheses describe which reported objects remain unseen. They are
        # internal exchangeable components, not sensor-readable identities.
        self.subsets = np.arange(1 << len(self.components), dtype=np.int64)
        self.membership = ((self.subsets[:, None] >> np.arange(len(self.components))) & 1).astype(bool)
        self.subset_probability = np.zeros(len(self.subsets))
        self.subset_probability[-1] = 1.
        self.initial_expected_count = len(self.components)+self.background_mass
        self.records = np.zeros((self.capacity, GAUSSIAN_FEATURES), dtype=np.float32)
        self.valid = np.zeros(self.capacity, dtype=bool)
        self.goals = np.zeros((self.capacity, 2))
        self._quadrature()
        # The background is a continuous product of observation likelihoods.
        # Values at integration/display points are caches of that function,
        # not independent cells. Arbitrary locations use the same history.
        self.background_history = []
        self._background_log_integral = np.zeros(self.integration_weights.shape)
        self._background_log_display = np.zeros(self.points.shape[:-1])
        self._background_log_normalizer = np.log((config.size-2*config.belief_search_margin)**2)
        self.local_background = (LocalBackground(config, self.integration_points,
                                 self.integration_weights, self.points*config.size)
                                 if config.belief_background_update == 'local' else None)
        self._background_exposure_log_display = np.zeros(self.points.shape[:-1])
        self.refresh(0)
        self.heat_scale = max(float(self.weights.max()), 1e-12)

    def _quadrature(self):
        c = self.c
        width = (c.size-2*c.belief_search_margin)/c.belief_grid
        lower = c.belief_search_margin + width*np.array(
            [(i % c.belief_grid, i // c.belief_grid) for i in range(self.cells)])
        self.bounds = np.stack((lower, lower+width), axis=1)
        nodes, weights = np.polynomial.legendre.leggauss(c.belief_integration_order)
        local = np.stack(np.meshgrid(nodes, nodes), -1).reshape(-1, 2)
        self.integration_points = lower[:, None]+(local+1)*width/2
        self.integration_weights = np.broadcast_to(
            np.outer(weights, weights).ravel()*(width/2)**2,
            self.integration_points.shape[:-1]).copy()
        # Raster is exclusively for replay rendering; its resolution cannot
        # change sensing, filtering, goals or actor inputs.
        self.display_side = 8
        axis = (np.arange(self.display_side)+.5)/self.display_side
        raster = np.stack(np.meshgrid(axis, axis), -1).reshape(-1, 2)
        self.points = (lower[:, None]+raster*width)/c.size

    @staticmethod
    def _density(component, points):
        delta = points-component['mean']
        covariance = component['covariance']
        inverse = np.linalg.inv(covariance)
        exponent = np.einsum('...i,ij,...j->...', delta, inverse, delta)
        return np.exp(-.5*exponent)/(2*np.pi*np.sqrt(np.linalg.det(covariance)))

    @property
    def remaining_types(self):
        result = np.zeros(3)
        for component in self.components:
            result[component['type']-1] += component['remaining']
        return result

    def update(self, detection_probability, observations, time):
        """Use only observed position/type to associate a new detection.

        Handles only deduplicate repeated sightings of already discovered
        objects; they never identify a report component or formation.
        """
        known = {track['handle']: track for track in self.tracks}
        for observation in observations:
            handle, position, typ, life = observation
            if handle not in known:
                if len(self.tracks) >= self.c.n_targets:
                    raise RuntimeError('Detected objects exceed configured capacity')
                association = self._associate(position, typ)
                variance = (self.c.position_noise/self.c.size)**2
                track = dict(handle=handle, mean=np.asarray(position)/self.c.size,
                             variance=variance, type=int(typ), life=int(life), seen=time,
                             initial_report_probability=association)
                self.tracks.append(track)
                known[handle] = track
            else:
                track = known[handle]
                variance = (self.c.position_noise/self.c.size)**2
                if variance == 0:
                    track['mean'] = np.asarray(position)/self.c.size
                    track['variance'] = 0.
                else:
                    gain = track['variance']/(track['variance']+variance)
                    track['mean'] += gain*(np.asarray(position)/self.c.size-track['mean'])
                    track['variance'] *= 1-gain
                track.update(type=int(typ), life=int(life), seen=time)

        # Integrate the continuous sensor likelihood on the allowed domain.
        # Apply changes in conditional moments to Gaussian parameters: using
        # moment differences avoids repeated boundary truncation drift when
        # the observation is uninformative. This is a bounded approximation.
        points = self.integration_points.reshape(-1, 2)
        pd = np.clip(detection_probability(points), 0, 1)
        log_survival = np.zeros(len(self.components))
        if np.any(pd):
            for index, component in enumerate(self.components):
                prior = self._density(component, points)*self.integration_weights.ravel()
                prior /= max(float(prior.sum()), np.finfo(float).tiny)
                posterior = prior*(1-pd)
                survival = max(float(posterior.sum()), 1e-15)
                log_survival[index] = np.log(survival)
                posterior /= survival
                prior_mean = prior@points
                posterior_mean = posterior@points
                prior_delta, posterior_delta = points-prior_mean, points-posterior_mean
                prior_cov = (prior_delta*prior[:, None]).T@prior_delta
                posterior_cov = (posterior_delta*posterior[:, None]).T@posterior_delta
                covariance = component['covariance']+posterior_cov-prior_cov
                values, vectors = np.linalg.eigh((covariance+covariance.T)/2)
                floor = self.c.belief_gaussian_min_sigma**2
                component['covariance'] = (vectors*np.maximum(values, floor))@vectors.T
                component['mean'] = np.clip(component['mean']+posterior_mean-prior_mean,
                                             self.c.belief_search_margin,
                                             self.c.size-self.c.belief_search_margin)
        log_likelihood = self.membership@log_survival
        supported = self.subset_probability > 0
        log_posterior = np.full(len(self.subsets), -np.inf)
        log_posterior[supported] = np.log(self.subset_probability[supported])+log_likelihood[supported]
        probability = np.exp(log_posterior-log_posterior.max())
        self.subset_probability = probability/probability.sum()
        for component, remaining in zip(self.components, self.subset_probability@self.membership):
            component['remaining'] = float(remaining)
        if self.c.belief_background_redistribute and np.any(pd):
            display_pd = detection_probability((self.points*self.c.size).reshape(-1, 2))
            self._background_exposure_log_display += self._log_miss(display_pd).reshape(self.points.shape[:-1])
            if self.local_background is not None:
                self.local_background.update(pd)
                self._background_log_normalizer = 0.
                self._background_log_integral = np.log(self.local_background.basis@self.local_background.mass).reshape(self.integration_weights.shape)
                self._background_log_display = np.log(self.local_background.display_basis@self.local_background.mass).reshape(self.points.shape[:-1])
            else:
                self.background_history.append(detection_probability)
                self._background_log_integral += self._log_miss(pd).reshape(self.integration_weights.shape)
                self._background_log_display += self._log_miss(display_pd).reshape(self.points.shape[:-1])
                peak = float(self._background_log_integral.max())
                self._background_log_normalizer = peak+np.log(float(
                    (np.exp(self._background_log_integral-peak)*self.integration_weights).sum()))
        self.refresh(time)

    @staticmethod
    def _log_miss(probability):
        return np.log1p(-np.clip(probability, 0., 1.-1e-12))

    def background_density(self, points):
        """Continuous spatial density, including the configured total mass.

        Likelihood callables must describe an immutable sensing opportunity.
        BeliefWorld supplies copied sensor poses for every update.
        """
        points = np.asarray(points, dtype=float)
        shape = points.shape[:-1]
        flat = points.reshape(-1, 2)
        if self.local_background is not None:
            inside = ((flat >= self.c.belief_search_margin)
                      & (flat <= self.c.size-self.c.belief_search_margin)).all(-1)
            return (self.background_mass*self.local_background.density(flat)*inside).reshape(shape)
        log_density = np.full(len(flat), -self._background_log_normalizer)
        for probability in self.background_history:
            log_density += self._log_miss(probability(flat))
        inside = ((flat >= self.c.belief_search_margin)
                  & (flat <= self.c.size-self.c.belief_search_margin)).all(-1)
        return (self.background_mass*np.exp(log_density)*inside).reshape(shape)

    def _associate(self, position, typ):
        """Marginalize report vs background and one-to-one report matching.

        Updating subset probabilities prevents multiple detections from
        consuming the same reported object, including simultaneous sightings.
        Use the background density before this sensing update for association.
        """
        background = float(self.background_density(np.asarray(position)))*self.background_type_prior[int(typ)-1]
        if self.c.belief_report_association == 'hard':
            matches = []
            for index, component in enumerate(self.components):
                if component['type'] != typ or component['remaining'] <= 0:
                    continue
                norm = float((self._density(component, self.integration_points)*self.integration_weights).sum())
                observed = dict(component, covariance=component['covariance']+np.eye(2)*self.c.position_noise**2)
                score = float(self._density(observed, np.asarray(position)))/max(norm, 1e-300)
                matches.append((score, -index))
            score, negative_index = max(matches, default=(-1., 0))
            if score <= background:
                return 0.
            index = -negative_index
            self.components[index]['remaining'] = 0.
            remaining = sum(1 << j for j, c in enumerate(self.components) if c['remaining'] > 0)
            self.subset_probability.fill(0)
            self.subset_probability[remaining] = 1.
            return 1.
        updated = self.subset_probability*background
        report_evidence = 0.
        for index, component in enumerate(self.components):
            if component['type'] != typ:
                continue
            normalizer = float((self._density(component, self.integration_points)
                                * self.integration_weights).sum())
            observed = dict(component, covariance=component['covariance']
                            + np.eye(2)*self.c.position_noise**2)
            likelihood = float(self._density(observed, np.asarray(position)))/max(normalizer, 1e-300)
            source = self.subsets[self.membership[:, index]]
            evidence = self.subset_probability[source]*likelihood
            updated[source ^ (1 << index)] += evidence
            report_evidence += float(evidence.sum())
        total = float(updated.sum())
        if total <= 0:
            raise ValueError('Detection is impossible under the configured report/background prior')
        self.subset_probability = updated/total
        return report_evidence/total

    def refresh(self, time):
        c = self.c
        self.records.fill(0)
        self.valid.fill(False)
        area = (c.size-2*c.belief_search_margin)**2
        measure = self.integration_weights
        background = (self.background_mass*np.exp(self._background_log_integral
                      - self._background_log_normalizer)*measure)
        types = np.zeros((*measure.shape, 3))
        raster_density = self.background_mass*np.exp(self._background_log_display-self._background_log_normalizer)
        for component in self.components:
            if component['remaining'] == 0:
                continue
            density = self._density(component, self.integration_points)
            normalizer = max(float((density*measure).sum()), np.finfo(float).tiny)
            types[..., component['type']-1] += density*measure/normalizer*component['remaining']
            raster_density += self._density(component, self.points*c.size)/normalizer*component['remaining']
        # Display weights are regenerated, not retained between observations.
        self.weights = raster_density*area/(self.cells*self.display_side**2)
        total = background+types.sum(-1)
        points = self.integration_points/c.size
        for k in range(self.cells):
            mass = float(total[k].sum())
            if mass <= 1e-12:
                continue
            probabilities = total[k]/mass
            mean = (probabilities[:, None]*points[k]).sum(0)
            delta = points[k]-mean
            covariance = (delta*probabilities[:, None]).T@delta
            type_mass = types[k].sum(0)
            reported = float(type_mass.sum())
            type_mass += float(background[k].sum())*self.background_type_prior
            fractions = type_mass/mass
            life = float(fractions@np.array([2., 1., 1.]))
            self.goals[k] = mean*c.size
            self.records[k] = [*mean, covariance[0, 0], covariance[0, 1], covariance[1, 1],
                               *fractions, life/2, mass, 0, 0, *mean,
                               float(background[k].sum()),
                               *((self.bounds[k, 1]-self.bounds[k, 0])/c.size), reported/mass]
            self.valid[k] = True
        for index, track in enumerate(self.tracks):
            k = self.cells+index
            typ = np.eye(3)[track['type']-1]
            self.records[k, :14] = [*track['mean'], track['variance'], 0, track['variance'],
                                    *typ, track['life']/2, 1, 1,
                                    (time-track['seen'])/c.horizon, *track['mean']]
            self.records[k, 17] = 1
            self.valid[k] = track['life'] > 0
            self.goals[k] = track['mean']*c.size

    def snapshot(self):
        return [dict(mean=x['mean'].tolist(), covariance=x['covariance'].tolist(),
                     type=x['type'], remaining=x['remaining']) for x in self.components]
