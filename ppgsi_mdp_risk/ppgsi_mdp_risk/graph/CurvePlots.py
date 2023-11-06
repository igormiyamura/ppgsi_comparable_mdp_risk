
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np, pandas as pd

class CurvePlots:
    def __init__(self, lib='matplotlib') -> None:
        self._lib = lib
        
    def plot_curve(self, dict_curves: dict, filter_prob: float=-1, less_equal=True, param_axis=None) -> None:
        """Plot curve

        Args:
            dict_curves (dict): dictionary with all the curves
            filter_prob (float, optional): filter less or greater than some probability (x axis). Defaults to -1.
            less_equal (bool, optional): True if less, False if greater. Defaults to True.
            param_axis (go.Figure, optional): Axis for subplots. Defaults to None.

        """        
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
        """Subplots of curves: (i) full x axis; (ii) less than `filter_prob`; (iii) greater than `filter_prob`

        Args:
            dict_curves (dict): dictionary with all the curves
            filter_prob (float, optional): filter less or greater than some probability (x axis). Defaults to -1.
        """        
        fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(18, 6))
        
        ax1 = self.plot_curve(dict_curves, -1, True, ax1)
        ax2 = self.plot_curve(dict_curves, filter_prob, True, ax2)
        ax3 = self.plot_curve(dict_curves, filter_prob, False, ax3)
        
        plt.show()
        
    def plot_curve_subplots_multiple_probabilities(self, dict_curves: dict, list_p: list) -> None:
        """Multiple probabilities subplots to verify the properties of each curve changing the
        initial probability.

        Args:
            dict_curves (dict): dictionary with all the curves
            list_p (list): list of all probabilities
        """
        fig, axs = plt.subplots(3, 3, figsize=(16,12))
        
        for risk_factor in dict_curves:
            c1, c2 = 0, 0
            for p in dict_curves[risk_factor]:
                curve = dict_curves[risk_factor][p]
                axs[c1, c2].plot(curve.keys(), curve.values())
            
                c1, c2 = self._define_counting(c1, c2, 3)
                
        c1, c2 = 0, 0
        for p in list_p:
            axs[c1, c2].axvline(x=p, color='green', linestyle='--')
            axs[c1, c2].set_title(f'Região ótima - Probabilidade [{np.round(p, 1)}]')
            c1, c2 = self._define_counting(c1, c2, 3)
            
    def _define_counting(self, c1, c2, max_c2=3):
        c2 += 1
        if c2 == max_c2:
            c2 = 0
            c1 += 1
            
        return c1, c2
    
    def plot_curve_from_dataframe(self, df: pd.DataFrame(), label: str, transpose: bool=False) -> None:
        df_plot = df.copy()
        fig, ax = plt.subplots()
        
        if transpose:
            df_plot = df_plot.T
            
        for c in df_plot.columns:
            curve = df_plot[c]
            if self._lib == 'matplotlib':
                ax.plot(curve.index, curve, label=f'{label}={c}')
                
        fig.legend(fontsize='xx-small')