
import matplotlib.pyplot as plt
import seaborn as sns  # For better color palette
import math

class MSSPlots:
    def __init__(self):
        pass

    def subplots_for_number_of_states(self, data, sharey=True, plot_only_first_state: bool = True):
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

        dict_values = {}
            
        for n in data.keys():
            if n not in dict_values:
                dict_values[n] = {}
            for prob in data[n].keys():
                for state, value in data[n][prob].items():
                    if (plot_only_first_state) and state != 0: continue
                    if state == 'sG': continue
                    
                    if state not in dict_values[n]:
                        dict_values[n][state] = []
                        
                    dict_values[n][state].append(value)

        for ax, (key, sub_dict), color in zip(axes, data.items(), colors):
            probabilities = list(sub_dict.keys())
            dict_values_for_n_states = dict_values[key]
                
            # Plot the line graph
            for state_value, values in dict_values_for_n_states.items():
                ax.plot(probabilities, values, marker='o', linewidth=2, markersize=6, label=f'n = {key}, s = {state_value}', color=color)
                
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
        
    def compare_data_and_analytical(self, data, data_analytical, sharey=True, plot_only_first_state: bool = True):
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

        dict_values = {}
        dict_values_analytical = {}

        for n in data.keys():
            if n not in dict_values:
                dict_values[n] = {}
            if n not in dict_values_analytical:
                dict_values_analytical[n] = {}
            for prob in data[n].keys():
                for state, value in data[n][prob].items():
                    if (plot_only_first_state) and state != 0: continue
                    if state == 'sG': continue

                    if state not in dict_values[n]:
                        dict_values[n][state] = []
                    if state not in dict_values_analytical[n]:
                        dict_values_analytical[n][state] = []

                    dict_values[n][state].append(value)
                    dict_values_analytical[n][state].append(data_analytical[n][prob][state])

        for ax, (key, sub_dict), color in zip(axes, data.items(), colors):
            probabilities = list(sub_dict.keys())
            dict_values_for_n_states = dict_values[key]
            dict_values_for_n_states_analytical = dict_values_analytical[key]

            # Plot the line graph for data
            for state_value, values in dict_values_for_n_states.items():
                ax.plot(probabilities, values, marker='o', alpha=0.5, linewidth=2, markersize=6, label=f'data: n = {key}, s = {state_value}', color=color)

            # Plot the line graph for data_analytical
            for state_value, values in dict_values_for_n_states_analytical.items():
                ax.plot(probabilities, values, marker='x', alpha=0.5, linewidth=2, markersize=6, linestyle='--', label=f'analytical: n = {key}, s = {state_value}', color='black')

            ax.set_title(f'Number of States => {key}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Probability (p)', fontsize=10)
            ax.grid(visible=True, linestyle='--', alpha=0.7)

            # Improve ticks
            ax.tick_params(axis='both', which='major', labelsize=9)
            ax.legend(fontsize=9, loc='upper right')

        # Remove unused subplots if any
        for i in range(len(data), len(axes)):
            fig.delaxes(axes[i])

        # Centralized y-axis label
        fig.text(0.04, 0.5, 'Value', va='center', rotation='vertical', fontsize=12, fontweight='bold')

        # Adjust layout for better spacing
        plt.tight_layout()
        plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9)  # Adjust spacing
        plt.show()