import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Define the length of the sequence
seq_length = 5

# Initialize a matrix to store the attention weights for each step
# At step i, token i can attend to tokens 0 to i (inclusive)
attention_weights_matrix = np.zeros((seq_length, seq_length))

# Simulate the attention weights for each token generation step
for i in range(seq_length):
    # Randomly generate attention weights for the current token towards all previous tokens (including itself)
    attention_weights_matrix[:i + 1, i] = np.random.rand(i + 1)

# Reverse the rows of the matrix to place the first token at the bottom
attention_weights_matrix = np.flipud(attention_weights_matrix)

# Create a heatmap for the attention weights matrix
plt.figure(figsize=(10, 6))
ax = sns.heatmap(attention_weights_matrix, cmap='Greys', annot=True, mask=attention_weights_matrix == 0, linewidths=.5, cbar=False)
plt.title('Attention Weights for Each Token Generation Step')
plt.xlabel('Generation Step')
plt.ylabel('Token Position')

# Adjust the ticks to properly align with the generation steps and token positions
ax.set_xticks(np.arange(0.5, seq_length + 0.5, 1))
ax.set_xticklabels(np.arange(1, seq_length + 1, 1))
ax.set_yticks(np.arange(0.5, seq_length + 0.5, 1))
ax.set_yticklabels(reversed(np.arange(1, seq_length + 1, 1)))  # Reverse the labels to match the row reversal

# Save the figure to a file
plt.savefig('pictures/attention_weights_over_steps_v6.png', dpi=300, bbox_inches='tight')

# Show the figure
plt.show()
