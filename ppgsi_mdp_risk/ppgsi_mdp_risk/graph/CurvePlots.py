
import matplotlib.pyplot as plt
import plotly.graph_objects as go

class CurvePlots:
    def __init__(self, lib='matplotlib') -> None:
        self._lib = lib
        
    def plot_curve(self, dict_curves: dict, filter_prob: float=-1, less_equal=True, param_axis=None) -> None:
        if param_axis is None:
            fig, ax = plt.subplots()
        
        for beta in dict_curves:
            curve = dict_curves[beta]
            
            if filter_prob != -1:
                if less_equal:
                    curve = {key: curve[key] for key in curve.keys() if key <= filter_prob}
                else:
                    curve = {key: curve[key] for key in curve.keys() if key >= filter_prob}
            
            if self._lib == 'matplotlib':
                if param_axis is None:
                    ax.plot(curve.keys(), curve.values(), label=f'beta={beta}')
                else:
                    param_axis.plot(curve.keys(), curve.values(), label=f'beta={beta}')
                
        if param_axis is None:
            fig.legend(loc='upper left')
            return fig, ax
        else:
            param_axis.legend(loc='upper left')
            return param_axis
                
    def plot_curve_subplots(self, dict_curves: dict, filter_prob: float=-1) -> None:
        fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(18, 6))
        
        ax1 = self.plot_curve(dict_curves, -1, True, ax1)
        ax2 = self.plot_curve(dict_curves, filter_prob, True, ax2)
        ax3 = self.plot_curve(dict_curves, filter_prob, False, ax3)
        
        plt.show()