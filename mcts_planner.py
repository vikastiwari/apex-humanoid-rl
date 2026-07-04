import numpy as np
import math
from typing import List, Tuple, Dict, Any

class MCTSNode:
    """
    A single node in the MCTS tree representing a kinodynamic state.
    """
    def __init__(self, position: np.ndarray, velocity: np.ndarray, parent=None, action=None):
        self.position = position  # 3D position (X, Y, Z)
        self.velocity = velocity  # 3D velocity (Vx, Vy, Vz)
        self.parent = parent
        self.action = action      # The chosen 3D target waypoint
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.value = 0.0

    @property
    def q_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value / self.visits


class MCTSPlanner:
    """
    High-Level kinodynamic MCTS Planner for Zero-Shot Humanoid Parkour.
    Operates as the 'Prefrontal Cortex', planning waypoint targets at ~2Hz.
    """
    def __init__(self, max_iterations: int = 100, v_max: float = 8.0, g: float = 9.81, uct_c: float = 1.414):
        self.max_iterations = max_iterations
        self.v_max = v_max  # Maximum velocity impulse (jump capability)
        self.g = g          # Gravity acceleration
        self.uct_c = uct_c  # Exploration constant

    def check_reachability(self, p_start: np.ndarray, v_start: np.ndarray, p_target: np.ndarray) -> bool:
        """
        Simplified ballistic point-mass physics function.
        Calculates if the robot can reach p_target using delta-v within max jump capacity.
        """
        dp = p_target - p_start
        times = np.linspace(0.1, 2.0, 20)  # Sample plausible flight times (0.1s to 2.0s)
        min_delta_v = float('inf')

        for t in times:
            v_tx = dp[0] / t
            v_ty = dp[1] / t
            v_tz = dp[2] / t + 0.5 * self.g * t
            
            v_takeoff = np.array([v_tx, v_ty, v_tz])
            delta_v = np.linalg.norm(v_takeoff - v_start)
            if delta_v < min_delta_v:
                min_delta_v = delta_v

        return min_delta_v <= self.v_max

    def select(self, node: MCTSNode) -> MCTSNode:
        """
        Selects a leaf node using the UCT (Upper Confidence Bound for Trees) formula.
        """
        current = node
        while len(current.children) > 0:
            # If any child is unvisited, return it immediately to expand
            unvisited = [c for c in current.children if c.visits == 0]
            if unvisited:
                return np.random.choice(unvisited)
            
            # Select child with highest UCT score
            log_parent_visits = math.log(current.visits)
            uct_scores = [
                child.q_value + self.uct_c * math.sqrt(log_parent_visits / child.visits)
                for child in current.children
            ]
            current = current.children[np.argmax(uct_scores)]
        return current

    def expand(self, node: MCTSNode, platforms: List[Dict[str, Any]]):
        """
        Expands the current node by generating child nodes representing reachable future waypoints.
        """
        # Candidate actions are the top surfaces of the next upcoming platforms
        candidates = []
        for plat in platforms:
            pos = np.array(plat['pos'])
            # Only search forward along the X-axis
            if pos[0] > node.position[0]:
                candidates.append(pos)
        
        # Sort candidates and keep nearest 3 to limit branching factor
        candidates.sort(key=lambda p: p[0])
        nearest_candidates = candidates[:3]

        for target in nearest_candidates:
            # Check kinodynamic reachability
            if self.check_reachability(node.position, node.velocity, target):
                # Transition: Point mass ballistic landing velocity estimation
                # Assume landing velocity roughly matches velocity required to drop onto target
                dx = target[0] - node.position[0]
                t_flight = dx / max(node.velocity[0], 1.0) # Est flight time
                v_land_z = (target[2] - node.position[2]) / t_flight - 0.5 * self.g * t_flight
                v_landing = np.array([node.velocity[0], 0.0, v_land_z])
                
                child = MCTSNode(
                    position=target,
                    velocity=v_landing,
                    parent=node,
                    action=target
                )
                node.children.append(child)

    def rollout(self, node: MCTSNode, platforms: List[Dict[str, Any]]) -> float:
        """
        Performs a rapid simulation rollout using point-mass ballistic checks.
        Returns a score representing the feasibility/progress of the rollout path.
        """
        current_pos = np.copy(node.position)
        current_vel = np.copy(node.velocity)
        steps = 0
        max_steps = 4
        total_distance = 0.0

        while steps < max_steps:
            # Find next platforms ahead
            candidates = [np.array(p['pos']) for p in platforms if p['pos'][0] > current_pos[0]]
            candidates.sort(key=lambda p: p[0])
            if not candidates:
                break
                
            # Try to jump to the nearest reachable platform
            target = candidates[0]
            if self.check_reachability(current_pos, current_vel, target):
                total_distance += (target[0] - current_pos[0])
                current_pos = target
                steps += 1
            else:
                # Target is too far, path terminates (fall)
                break
                
        # Reward is proportional to forward distance traversed
        return total_distance

    def backpropagate(self, node: MCTSNode, reward: float):
        """
        Backpropagates the rollout score up to the root node.
        """
        current = node
        while current is not None:
            current.visits += 1
            current.value += reward
            current = current.parent

    def plan(self, current_pos: np.ndarray, current_vel: np.ndarray, platforms: List[Dict[str, Any]]) -> np.ndarray:
        """
        Executes MCTS planning iterations and returns the optimal next 3D waypoint.
        """
        root = MCTSNode(current_pos, current_vel)
        
        for _ in range(self.max_iterations):
            # 1. Selection
            leaf = self.select(root)
            
            # 2. Expansion
            if leaf.visits > 0 or leaf == root:
                self.expand(leaf, platforms)
                if leaf.children:
                    leaf = leaf.children[0]
            
            # 3. Rollout / Simulation
            reward = self.rollout(leaf, platforms)
            
            # 4. Backpropagation
            self.backpropagate(leaf, reward)

        # Return the action of the child with the highest visit count (most robust selection)
        if not root.children:
            # Fallback: if no platform is reachable, project a forward target
            return current_pos + np.array([1.0, 0.0, 0.0])
            
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action


# ==============================================================================
# ASYNCHRONOUS SYSTEM ARCHITECTURE DOCUMENTATION & IMPLEMENTATION NOTES
# ==============================================================================
# In a production deployment of Project APEX, the system utilizes a Dual-Rate 
# Asynchronous execution architecture:
#
# 1. High-Level Prefrontal Cortex (MCTS) - CPU Loop @ 2Hz (Every 500ms):
#    - Subscribes to the robot's state estimator (CoM Position, CoM Velocity).
#    - Resolves the environment's static map representation (platforms, gaps).
#    - Computes the MCTS search tree, outputting the next optimal 3D target waypoint.
#    - Publishes the target waypoint to a thread-safe shared state buffer.
#
# 2. Low-Level Joint Controller (PPO Policy) - GPU/MJX Loop @ 1000Hz (Every 1ms):
#    - Subscribes to high-frequency proprioceptive joint angles and foot tactile sensors.
#    - Reads the latest target waypoint asynchronously from the shared state buffer.
#    - Calculates the relative vector (target_waypoint - pelvis_position).
#    - Evaluates the neural network policy to command joint motors.
#
# Example Multi-Threaded Scaffold:
#
# import threading
# import time
#
# shared_waypoint = np.zeros(3)
# lock = threading.Lock()
#
# def mcts_planning_thread(platforms):
#     planner = MCTSPlanner()
#     while running:
#         state_pos, state_vel = get_robot_state()
#         next_waypoint = planner.plan(state_pos, state_vel, platforms)
#         with lock:
#             global shared_waypoint
#             shared_waypoint = next_waypoint
#         time.sleep(0.5)  # Run at 2 Hz
#
# def ppo_control_thread():
#     while running:
#         with lock:
#             target = np.copy(shared_waypoint)
#         action = ppo_policy.step(obs, target)
#         send_motor_commands(action)
#         time.sleep(0.001)  # Run at 1000 Hz
# ==============================================================================
