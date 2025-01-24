
import matplotlib.pyplot as plt
import seaborn as sns  # For better color palette
import math

class MSSPlots:
    def __init__(self):
        pass

    def subplots_for_number_of_states(self, data, sharey=True):
        # Use a color palette
        colors = sns.color_palette("tab10", len(data))

        # Initialize figure and subplots
        n = len(data)
        if n > 4:
            # Calculate rows and columns dynamically
            cols = min(n, 4)  # Maximum of 4 columns
            rows = math.ceil(n / cols)  # Rows depend on the number of graphs
        else:
            rows, cols = 1, n  # Single row layout for <= 4 graphs

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), sharey=sharey)

        # Flatten axes for easy iteration if multi-row layout
        if n > 1:
            axes = axes.flatten()
        else:
            axes = [axes]  # Ensure axes is a list even for a single subplot

        for ax, (key, sub_dict), color in zip(axes, data.items(), colors):
            probabilities = list(sub_dict.keys())
            values = list(sub_dict.values())
            
            # Plot the line graph
            ax.plot(probabilities, values, marker='o', linewidth=2, markersize=6, label=f'n = {key}', color=color)
            ax.set_title(f'Number of States => {key}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Probability (p)', fontsize=10)
            ax.grid(visible=True, linestyle='--', alpha=0.7)
            
            # Improve ticks
            ax.tick_params(axis='both', which='major', labelsize=9)
            ax.legend(fontsize=9, loc='upper left')

        # Remove unused subplots if any
        for i in range(len(data), len(axes)):
            fig.delaxes(axes[i])

        # Centralized y-axis label
        fig.text(0.04, 0.5, 'Value', va='center', rotation='vertical', fontsize=12, fontweight='bold')

        # Adjust layout for better spacing
        plt.tight_layout()
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9)  # Adjust spacing
        plt.show()