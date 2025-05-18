import numpy as np
from collections import deque
import wandb
import time
from tqdm import tqdm
import os
import torch

import market.synthetic_chriss_almgren as sca
from agent.ddpg import Agent

# Create simulation environment
env = sca.MarketEnvironment()

# Defining the agent
# Initialize Feed-forward DNNs for Actor and Critic models. 
agent = Agent(
    state_size=env.observation_space_dimension(), 
    action_size=env.action_space_dimension(), 
    random_seed=0
)

lqt = 60        # Set the liquidation time
n_trades = 60   # Set the number of trades
tr = 1e-6       # Set trader's risk aversion
episodes = 10000    # Set the number of episodes to run the simulation

def checkpoint(i_episode, scores_window, scores):
    print("\rEpisode {}\t\tAvg Score: {:.2f}\t\tMax Score: {:.2f}".format(i_episode, np.mean(scores_window), np.max(scores)), end="")
    if i_episode % 100 == 0:
        print("\rEpisode {}\t\tAvg Score: {:.2f}".format(i_episode, np.mean(scores_window)))
    if i_episode % 200 == 0:
        torch.save(agent.actor_local.state_dict(), os.path.join(checkpoint_dir, f'checkpoint_{i_episode}_actor.pth'))
        torch.save(agent.critic_local.state_dict(), os.path.join(checkpoint_dir, f'checkpoint_{i_episode}_critic.pth'))

# 
if __name__ == "__main__":
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = "model_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize wandb
    run = wandb.init(
        project="optimal-trade-execution",
        config={
            "liquidation_time": lqt,
            "num_trades": n_trades,
            "risk_aversion": tr,
            "episodes": episodes,
            "algorithm": "DDPG"
        }
    )
    
    # Pre-allocate numpy array for better performance
    shortfall_hist = np.zeros(episodes)
    shortfall_deque = deque(maxlen=100)
    scores = []  # Store all scores for max score calculation
    
    # Use tqdm for better progress tracking
    pbar = tqdm(range(episodes), desc="Training")
    best_avg_shortfall = float('inf')
    start_time = time.time()
    
    for episode in pbar: 
        # Reset the environment
        cur_state = env.reset(
            seed=episode, 
            liquid_time=lqt, 
            num_trades=n_trades, 
            lamb=tr
        )

        # set the environment to make transactions
        env.start_transactions()
        
        episode_reward = 0
        
        for i in range(n_trades + 1):
            # Predict the best action for the current state
            action = agent.act(cur_state, add_noise=True)
            
            # Action is performed and new state, reward, info are received
            new_state, reward, done, info = env.step(action)
            
            # current state, action, reward, new state are stored in the experience replay
            agent.step(cur_state, action, reward, new_state, done)
            
            # roll over new state
            cur_state = new_state
            
            episode_reward += reward
            
            if info.done:
                # Store directly in pre-allocated array - more efficient
                shortfall_hist[episode] = info.implementation_shortfall
                shortfall_deque.append(info.implementation_shortfall)
                scores.append(episode_reward)  # Add to scores list
                
                # Calculate metrics
                avg_shortfall = np.mean(shortfall_deque)
                
                # Call checkpoint function
                checkpoint(episode, shortfall_deque, scores)
                
                # Log episode results to wandb
                wandb.log({
                    "episode": episode,
                    "implementation_shortfall": info.implementation_shortfall,
                    "episode_reward": episode_reward,
                    "avg_shortfall_100": avg_shortfall,
                    "training_time": time.time() - start_time
                })
                
                # Update progress bar with current metrics
                pbar.set_postfix({
                    'avg_shortfall': f'${avg_shortfall:.2f}',
                    'reward': f'{episode_reward}'
                })
                
                # Save best model based on performance
                if avg_shortfall < best_avg_shortfall and episode > 100:
                    best_avg_shortfall = avg_shortfall
                    actor_path = os.path.join(checkpoint_dir, "best_actor.pth")
                    critic_path = os.path.join(checkpoint_dir, "best_critic.pth")
                    torch.save(agent.actor_local.state_dict(), actor_path)
                    torch.save(agent.critic_local.state_dict(), critic_path)
                    wandb.log({"best_avg_shortfall": best_avg_shortfall})
                
                break
        
        # Regular checkpointing
        if (episode + 1) % 100 == 0:
            # Save periodically with episode number
            actor_path = os.path.join(checkpoint_dir, f"actor_episode_{episode+1}.pth")
            critic_path = os.path.join(checkpoint_dir, f"critic_episode_{episode+1}.pth")
            
            # # Log files to wandb
            # wandb.save(actor_path)
            # wandb.save(critic_path)
            
            # Log recent performance
            wandb.log({
                "checkpoint_episode": episode + 1,
                "checkpoint_avg_shortfall": np.mean(shortfall_deque)
            })
    
    # Final evaluation metrics
    final_avg_shortfall = np.mean(shortfall_hist)
    wandb.log({
        "final_avg_shortfall": final_avg_shortfall,
        "total_training_time": time.time() - start_time
    })
    
    # Save final model
    final_actor_path = os.path.join(checkpoint_dir, "final_actor.pth")
    final_critic_path = os.path.join(checkpoint_dir, "final_critic.pth")
#     agent.save(final_actor_path, final_critic_path)
#     wandb.save(final_actor_path)
#     wandb.save(final_critic_path)
    
    # Finish wandb run
    wandb.finish()
    
    print(f'Training complete. Average Implementation Shortfall: ${final_avg_shortfall:,.2f}')
    print(f'Best Average Shortfall (100 ep window): ${best_avg_shortfall:,.2f}')