import numpy as np
from easy_21 import State, Action, init_game, step
import matplotlib.pyplot as plt


class MCControl:
    def __init__(self, n_episodes: int = 10000):
        self.n_episodes = n_episodes
        self.V = {}  # State -> float
        self.Q = {}  # (State, Action) -> float 
        self.returns = {}  # (State, Action) -> list[float]
        self.policy = {}  # State -> Action
        self.N_state = {}  # State -> int
        self.N_state_action = {}  # (State, Action) -> int
        self.N_0 = 100

    def get_state_key(self, state: State) -> tuple:
        """Convert state to hashable tuple"""
        return (state.player_sum, state.dealer_sum)

    def get_legal_actions(self) -> list[Action]:
        """Return list of legal actions"""
        return [Action("hit"), Action("stick")]

    def get_action(self, state: State, epsilon: float) -> Action:
        """Epsilon-greedy action selection"""
        state_key = self.get_state_key(state)

        if state_key not in self.policy or np.random.random() < epsilon:
            return np.random.choice(self.get_legal_actions())

        return self.policy[state_key]

    def update_value_function(self, episode: list[tuple[State, Action, float]]):
        """Update value function using every-visit MC with step size alpha"""
        G = 0
        for state, action, reward in reversed(episode):
            G = G + reward
            state_key = self.get_state_key(state)
            self.N_state[state_key] = self.N_state.get(state_key, 0) + 1
            state_action = (state_key, action.action)
            self.N_state_action[state_action] = self.N_state_action.get(state_action, 0) + 1

            if state_action not in self.Q:
                self.Q[state_action] = 0

            # Update Q value using step size alpha
            self.Q[state_action] += 1/self.N_state_action[state_action] * (G - self.Q[state_action])
            self.V[state_key] = max(self.Q.get((state_key, action.action), 0) for action in self.get_legal_actions())

        # Update policy
        best_action = max(self.get_legal_actions(), 
                         key=lambda a: self.Q.get((state_key, a.action), 0))
        self.policy[state_key] = best_action

    def generate_episode(self) -> list[tuple[State, Action, float]]:
        """Generate one episode using current policy"""
        episode = []
        state = init_game()
        
        while True:
            epsilon = self.N_0 / (self.N_0 + self.N_state.get(self.get_state_key(state), 0))
            action = self.get_action(state, epsilon)
            old_state = state
            state, reward = step(state, action)
            episode.append((old_state, action, reward.reward))
            
            if reward.reward != 0:  # Episode ended
                break
            
        return episode
    
    def run(self):
        for _ in range(self.n_episodes):
            episode = self.generate_episode()
            self.update_value_function(episode)

    def plot_value_function(self):
        # Create meshgrid of all possible states
        player_sums = np.arange(1, 22)
        dealer_sums = np.arange(1, 11)
        X, Y = np.meshgrid(player_sums, dealer_sums)
        
        # Get values for each state
        Z = np.zeros_like(X, dtype=float)
        for i, dealer_sum in enumerate(dealer_sums):
            for j, player_sum in enumerate(player_sums):
                state_key = (player_sum, dealer_sum)
                Z[i, j] = self.V.get(state_key, 0)
        
        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot surface
        surface = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        
        # Add labels and colorbar
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Sum')
        ax.set_zlabel('Value')
        ax.set_title('Value Function')
        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
        
        plt.show()

    def plot_value_heatmap(self):
        # Create arrays for all possible states
        player_sums = np.arange(1, 22)
        dealer_sums = np.arange(1, 11)
        
        # Get values for each state
        Z = np.zeros((len(dealer_sums), len(player_sums)))
        for i, dealer_sum in enumerate(dealer_sums):
            for j, player_sum in enumerate(player_sums):
                state_key = (player_sum, dealer_sum)
                Z[i, j] = self.V.get(state_key, 0)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        plt.imshow(Z, cmap='viridis', aspect='auto')
        plt.colorbar(label='Value')
        
        # Add labels and ticks
        plt.xlabel('Player Sum')
        plt.ylabel('Dealer Sum')
        plt.title('Value Function Heatmap')
        
        # Set tick labels
        plt.xticks(range(len(player_sums)), player_sums)
        plt.yticks(range(len(dealer_sums)), dealer_sums)
        
        plt.show()

    def plot_optimal_policy(self):
        # Create arrays for all possible states
        player_sums = np.arange(1, 22)
        dealer_sums = np.arange(1, 11)
        
        # Create figure and axis with larger size
        plt.figure(figsize=(15, 8))
        
        # Create policy matrix
        policy_matrix = np.zeros((len(dealer_sums), len(player_sums)), dtype=str)
        for i, dealer_sum in enumerate(dealer_sums):
            for j, player_sum in enumerate(player_sums):
                state_key = (player_sum, dealer_sum)
                if state_key in self.policy:
                    policy_matrix[i, j] = 'H' if self.policy[state_key].action == 'hit' else 'S'
                else:
                    policy_matrix[i, j] = ''
        
        # Create table
        table = plt.table(
            cellText=policy_matrix,
            cellColours=np.where(policy_matrix == 'H', '#ffcccc', '#cce5ff'),
            cellLoc='center',
            loc='center',
            rowLabels=dealer_sums,
            colLabels=player_sums,
        )
        
        # Customize table appearance
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Add labels and title
        plt.title('Optimal Policy (H: Hit, S: Stick)\nDealer Sum vs Player Sum', pad=50)
        plt.ylabel('Dealer Sum', labelpad=20)
        plt.xlabel('Player Sum', labelpad=20)
        
        # Remove axis
        plt.axis('off')
        
        plt.show()


def main():
    simulation = MCControl(1000000)
    simulation.run()
    simulation.plot_value_function()
    simulation.plot_value_heatmap()
    simulation.plot_optimal_policy()


if __name__ == "__main__":
    main()
