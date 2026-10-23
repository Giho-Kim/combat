"""Per-agent task automaton for belief mode.

Two orthogonal regions describe one agent:

* motion: IDLE -> TRANSIT -> {ARRIVED, SWEEPING}, driven by executor events.
* engagement: the track an agent is locked into by an ongoing strike, driven
  by the hidden world and synchronized as observed LOCK/RELEASE events.

The exported phase is hierarchical: INACTIVE overrides ENGAGED, which
overrides the motion state. A commitment register (held component and the step
until which it is held) records the last decision. The action mask is a pure
function of these states and the public belief.

Decision rules: 'interval' re-decides every `hold_until` steps (the original
behavior). 'event' is a semi-MDP: an agent re-decides when its task terminates
(a termination event since its last decision, or an interrupt), with
`hold_until` as an upper bound.
"""
from enum import Enum, IntEnum

import numpy as np


class TaskPhase(IntEnum):
    INACTIVE = 0
    IDLE = 1      # no executable component
    TRANSIT = 2   # following a path toward a component's destination
    SWEEPING = 3  # lawnmower passes over a selected region
    ARRIVED = 4   # holding at a reached destination
    ENGAGED = 5   # locked into an ongoing task on a track


MOTION_PHASES = (TaskPhase.IDLE, TaskPhase.TRANSIT, TaskPhase.SWEEPING, TaskPhase.ARRIVED)


class TaskEvent(Enum):
    SELECT = 'select'          # executor starts on a new component
    RETARGET = 'retarget'      # destination of the same component moved
    ARRIVE = 'arrive'          # point destination reached
    SWEEP = 'sweep'            # region entry reached; passes begin
    LEG_DONE = 'leg_done'      # one lawnmower waypoint reached
    PASS_DONE = 'pass_done'    # one lawnmower pass completed; another starts
    INVALIDATE = 'invalidate'  # selected component is no longer valid
    LOCK = 'lock'
    RELEASE = 'release'
    DEACTIVATE = 'deactivate'


P, E = TaskPhase, TaskEvent
MOTION_TRANSITIONS = {
    **{(phase, E.SELECT): P.TRANSIT for phase in MOTION_PHASES},
    **{(phase, E.INVALIDATE): P.IDLE for phase in MOTION_PHASES},
    (P.TRANSIT, E.RETARGET): P.TRANSIT,
    (P.ARRIVED, E.RETARGET): P.TRANSIT,
    (P.TRANSIT, E.ARRIVE): P.ARRIVED,
    (P.TRANSIT, E.SWEEP): P.SWEEPING,
    (P.SWEEPING, E.LEG_DONE): P.SWEEPING,
    (P.SWEEPING, E.PASS_DONE): P.SWEEPING,
}
del P, E

# A full lawnmower pass takes most of an episode (~1270 of 1800 steps), so the
# default ends a search task after each sweep leg (~30-70 steps), not a pass.
DEFAULT_TERMINATIONS = frozenset((TaskEvent.ARRIVE, TaskEvent.LEG_DONE, TaskEvent.INVALIDATE,
                                  TaskEvent.RELEASE))
DECISION_RULES = ('interval', 'event')


class TaskAutomata:
    """Task automata of all agents in one episode."""
    def __init__(self, n, terminations=DEFAULT_TERMINATIONS):
        self.n = n
        self.terminations = frozenset(terminations)
        self.active = np.ones(n, dtype=bool)
        self.motion = np.full(n, TaskPhase.IDLE, dtype=int)
        self.component = np.full(n, -1, dtype=int)   # executor's component
        self.engaged = np.full(n, -1, dtype=int)     # locked track component
        self.held = np.full(n, -1, dtype=int)        # last decided component
        self.hold_until = np.zeros(n, dtype=int)
        self.due = np.zeros(n, dtype=bool)            # task terminated since the last decision
        self.events = []                              # (agent, event, component) this step

    def fire(self, i, event, component=None):
        if not self.active[i]:
            raise ValueError(f'Agent {i} is inactive; {event.value} is undefined')
        if event is TaskEvent.DEACTIVATE:
            self.active[i] = False
            self.engaged[i] = -1
        elif event is TaskEvent.LOCK:
            self.engaged[i] = component
        elif event is TaskEvent.RELEASE:
            self.engaged[i] = -1
        else:
            key = (TaskPhase(self.motion[i]), event)
            if key not in MOTION_TRANSITIONS:
                raise ValueError(f'Agent {i}: no transition for {event.value} in {key[0].name}')
            self.motion[i] = MOTION_TRANSITIONS[key]
            if event is TaskEvent.SELECT:
                self.component[i] = component
            elif event is TaskEvent.INVALIDATE:
                self.component[i] = -1
        if event in self.terminations:
            self.due[i] = True
        self.events.append((i, event, component))

    def commit(self, decision, components, until):
        self.held[decision] = components[decision]
        self.hold_until[decision] = until
        self.due[decision] = False

    def interrupt(self):
        """New shared information ends every free agent's current task."""
        self.due |= self.active & (self.engaged < 0)

    def sync(self, active, locks):
        """Turn world-side changes into events, after transitions and sensing."""
        for i in np.flatnonzero(self.active):
            if locks[i] != self.engaged[i]:
                if locks[i] >= 0:
                    self.fire(i, TaskEvent.LOCK, int(locks[i]))
                else:
                    self.fire(i, TaskEvent.RELEASE)
            if not active[i]:
                self.fire(i, TaskEvent.DEACTIVATE)

    def phases(self):
        return np.where(~self.active, TaskPhase.INACTIVE,
                        np.where(self.engaged >= 0, TaskPhase.ENGAGED, self.motion))

    def sweeping(self, valid):
        """Region claimed by each agent's sweep, or -1."""
        claim = self.active & (self.motion == TaskPhase.SWEEPING)
        claim &= valid[np.maximum(self.component, 0)]
        return np.where(claim, self.component, -1)

    def action_mask(self, valid, track_life, cells, t, *, initial_intel, commit_target,
                    decision_rule='interval'):
        """Allowed components and decision points; `track_life[k]` is the
        expected remaining participants needed by track slot `cells+k`."""
        if decision_rule not in DECISION_RULES:
            raise ValueError(f'Unknown decision rule {decision_rule}')
        n = self.n
        mask = np.broadcast_to(valid, (n, len(valid))).copy()
        locks, sweeping = self.engaged, self.sweeping(valid)
        # A sweep claims its region exclusively.
        for i in np.flatnonzero(sweeping >= 0):
            mask[:, sweeping[i]] = False
            mask[i, sweeping[i]] = True
        # A track admits no new participants once enough are locked in.
        for k, life in enumerate(track_life):
            j = cells + k
            if (locks == j).sum() >= life:
                mask[:, j] = locks == j
        decision = self.active & (locks < 0)
        for i in range(n):
            held = self.held[i]
            if locks[i] >= 0:
                mask[i] = False
                mask[i, locks[i]] = True
            elif held >= 0 and mask[i, held] and (
                    commit_target or (t < self.hold_until[i] and not self.due[i])
                    if decision_rule == 'event' else
                    (sweeping[i] >= 0 and not initial_intel) or commit_target or t < self.hold_until[i]):
                mask[i] = False
                mask[i, held] = True
                decision[i] = False
            if not self.active[i] or not mask[i].any():
                mask[i] = False
                mask[i, 0] = True  # Finite categorical; invalid component is an idle action.
                decision[i] = False
        return mask, decision
