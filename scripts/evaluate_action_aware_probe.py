import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import pointmass_rl.graph_critic as graph
Original=graph.GraphQBody
class SelectedGraph(Original):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.assignment_encoder=torch.nn.Linear(6,64);self.drone_encoder.register_forward_hook(self.add_assignment)
 def add_assignment(self,module,inputs,output):return output+self.assignment_encoder(self.assigned_target)
 def forward(self,state,joint_action):
  n,t=self.dimensions(state,joint_action);target=state[...,1+4*n:1+4*n+6*t].reshape(*state.shape[:-1],t,6);self.assigned_target=target.gather(-2,joint_action.unsqueeze(-1).expand(*joint_action.shape,6));return super().forward(state,joint_action)
graph.GraphQBody=SelectedGraph
sys.argv=['evaluation','runs/critic_2x2_action_aware_self']
exec(compile(Path('scripts/evaluate_2x2_failures.py').read_text(),'evaluation','exec'))
