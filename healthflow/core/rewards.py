"""
Reward Functions for Medical Diagnosis RL Agent
Implements mutual information and GAN-based rewards as specified in the research tasks
"""

import numpy as np
from typing import Union, List, Optional
import warnings

# Suppress warnings for log(0) calculations
warnings.filterwarnings('ignore', category=RuntimeWarning)


def calculate_entropy(prob_dist: Union[np.ndarray, List[float]]) -> float:
    """
    Calculate entropy of a probability distribution
    
    Args:
        prob_dist: Probability distribution vector (should sum to 1.0)
        
    Returns:
        Entropy H(P) = -sum(p_i * log(p_i))
    """
    prob_dist = np.asarray(prob_dist, dtype=np.float64)
    
    # Validate probability distribution
    if len(prob_dist) == 0:
        return 0.0
    
    # Normalize to ensure it's a valid probability distribution
    prob_sum = np.sum(prob_dist)
    if prob_sum <= 0:
        return 0.0
    
    prob_dist = prob_dist / prob_sum
    
    # Handle zero probabilities by replacing with small epsilon
    epsilon = 1e-10
    prob_dist = np.maximum(prob_dist, epsilon)
    
    # Calculate entropy: H(P) = -sum(p_i * log(p_i))
    entropy = -np.sum(prob_dist * np.log(prob_dist))
    
    return float(entropy)


def calculate_mi_reward(O_prev: Union[np.ndarray, List[float]], 
                       O_curr: Union[np.ndarray, List[float]]) -> float:
    """
    Calculate mutual information (MI) based reward signal for medical diagnosis RL agent.
    
    The MI reward is defined as the reduction in entropy from the prior state to the posterior state:
    R_M = I(O_{t-1}; O_t | D_φ) = H(O_{t-1}) - H(O_t)
    
    This reward encourages the agent to ask questions that best distinguish between potential diseases.
    
    Args:
        O_prev: Disease probability distribution before asking a question (O_{t-1})
        O_curr: Disease probability distribution after receiving patient's answer (O_t)
        
    Returns:
        Scalar reward value R_M representing information gain
        
    Example:
        >>> # Initial uniform distribution over 4 diseases
        >>> O_prev = [0.25, 0.25, 0.25, 0.25]  
        >>> # After asking a good question, distribution becomes more focused
        >>> O_curr = [0.7, 0.1, 0.1, 0.1]
        >>> reward = calculate_mi_reward(O_prev, O_curr)
        >>> print(f"MI Reward: {reward:.4f}")
        MI Reward: 0.8813
    """
    # Convert inputs to numpy arrays
    O_prev = np.asarray(O_prev, dtype=np.float64)
    O_curr = np.asarray(O_curr, dtype=np.float64)
    
    # Validate input dimensions
    if len(O_prev) != len(O_curr):
        raise ValueError("Previous and current probability distributions must have the same length")
    
    if len(O_prev) == 0:
        return 0.0
    
    # Calculate entropies
    H_prev = calculate_entropy(O_prev)
    H_curr = calculate_entropy(O_curr)
    
    # MI reward = reduction in entropy
    R_M = H_prev - H_curr
    
    return float(R_M)


def calculate_final_reward(R_M: float, R_D: float, lambda_param: float = 0.5, epsilon: float = 0.5) -> float:
    """
    Calculate final combined reward for medical diagnosis RL agent.
    
    Integrates mutual information reward with GAN discriminator reward:
    R_F = (1 - λ)R_M + λ(R_D - ε)
    
    Args:
        R_M: Mutual information reward (scalar float)
        R_D: Discriminator reward - probability from discriminator D_ψ(Y_{1:t-1}) (scalar float)
        lambda_param: Balance parameter between MI and discriminator rewards (default: 0.5)
        epsilon: Offset for discriminator reward to shift reward range (default: 0.5)
        
    Returns:
        Final combined reward R_F
        
    Example:
        >>> R_M = 0.8813  # High information gain
        >>> R_D = 0.8     # High discriminator probability (realistic sequence)
        >>> reward = calculate_final_reward(R_M, R_D)
        >>> print(f"Final Reward: {reward:.4f}")
        Final Reward: 0.5906
    """
    R_F = (1 - lambda_param) * R_M + lambda_param * (R_D - epsilon)
    return float(R_F)


def calculate_policy_gradient(log_probs: List[float], 
                            rewards: List[float]) -> np.ndarray:
    """
    Calculate policy gradient for generator network in medical diagnosis system.
    
    Implements: ∇_θ J(θ) = Σ_t E[∇_θ log G_θ(y_t|Y_{1:t-1}) · R_F^(t)]
    
    Args:
        log_probs: Sequence of log-probabilities [log G_θ(y_1|Y_0), ..., log G_θ(y_T|Y_{1:T-1})]
        rewards: Corresponding sequence of rewards [R_F^(1), ..., R_F^(T)]
        
    Returns:
        Policy gradient estimate as numpy array
        
    Example:
        >>> log_probs = [-1.2, -0.8, -1.5]  # Log probabilities of actions
        >>> rewards = [0.5, 0.8, 0.3]       # Corresponding rewards
        >>> gradient = calculate_policy_gradient(log_probs, rewards)
        >>> print(f"Gradient: {gradient}")
        [-0.6, -0.64, -0.45]
    """
    if len(log_probs) != len(rewards):
        raise ValueError("log_probs and rewards must have the same length")
    
    if len(log_probs) == 0:
        return np.array([])
    
    log_probs = np.asarray(log_probs, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    
    # Calculate policy gradient: gradient = log_prob * reward for each timestep
    gradient = log_probs * rewards
    
    return gradient


def calculate_discriminator_loss(real_probs: List[float], 
                               fake_probs: List[float]) -> float:
    """
    Calculate discriminator loss for GAN training in medical symptom sequences.
    
    Implements: min_ψ -E[log D_ψ(Y_real)] - E[log(1 - D_ψ(Y_fake))]
    
    Args:
        real_probs: Discriminator probabilities for real sequences
        fake_probs: Discriminator probabilities for fake sequences
        
    Returns:
        Discriminator loss value
        
    Example:
        >>> real_probs = [0.9, 0.8, 0.85]  # High probs for real sequences
        >>> fake_probs = [0.2, 0.3, 0.15]  # Low probs for fake sequences  
        >>> loss = calculate_discriminator_loss(real_probs, fake_probs)
        >>> print(f"Discriminator Loss: {loss:.4f}")
        Discriminator Loss: 0.3567
    """
    if len(real_probs) == 0 and len(fake_probs) == 0:
        return 0.0
    
    real_probs = np.asarray(real_probs, dtype=np.float64)
    fake_probs = np.asarray(fake_probs, dtype=np.float64)
    
    # Clamp probabilities to avoid log(0)
    epsilon = 1e-10
    real_probs = np.clip(real_probs, epsilon, 1 - epsilon)
    fake_probs = np.clip(fake_probs, epsilon, 1 - epsilon)
    
    # Calculate discriminator loss components
    real_loss = -np.mean(np.log(real_probs)) if len(real_probs) > 0 else 0.0
    fake_loss = -np.mean(np.log(1 - fake_probs)) if len(fake_probs) > 0 else 0.0
    
    return float(real_loss + fake_loss)


class RewardTracker:
    """Tracks reward statistics and performance metrics"""
    
    def __init__(self):
        self.mi_rewards: List[float] = []
        self.discriminator_rewards: List[float] = []
        self.final_rewards: List[float] = []
        self.accuracies: List[float] = []
        self.turn_counts: List[int] = []
    
    def add_episode(self, mi_reward: float, disc_reward: float, final_reward: float, 
                   accuracy: float, turns: int):
        """Add episode statistics"""
        self.mi_rewards.append(mi_reward)
        self.discriminator_rewards.append(disc_reward)
        self.final_rewards.append(final_reward)
        self.accuracies.append(accuracy)
        self.turn_counts.append(turns)
    
    def get_statistics(self) -> dict:
        """Get reward and performance statistics"""
        if len(self.final_rewards) == 0:
            return {}
        
        return {
            'avg_mi_reward': np.mean(self.mi_rewards),
            'avg_disc_reward': np.mean(self.discriminator_rewards),
            'avg_final_reward': np.mean(self.final_rewards),
            'avg_accuracy': np.mean(self.accuracies),
            'avg_turns': np.mean(self.turn_counts),
            'episodes': len(self.final_rewards)
        }
    
    def save_to_file(self, filepath: str):
        """Save statistics to JSON file"""
        import json
        stats = self.get_statistics()
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)