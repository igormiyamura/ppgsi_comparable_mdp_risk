
# Matplotlib Imports
import matplotlib.pyplot as plt

# Plotly Imports
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Other Imports
import numpy as np, pandas as pd

class CurvePlots:
    def __init__(self, lib='matplotlib') -> None:
        self._lib = lib
        self._colors = {
            'EXP (inf)': 'blue',
            'EXP (sup)': 'magenta',
            'PWL (inf)': 'green',
            'PWL (sup)': 'black',
            'VAR (inf)': 'orange',
            'VAR (sup)': 'red',
            'CVAR (inf)': 'maroon',
            'CVAR (sup)': 'olive'
        }
        
    def plot_curve(self, dict_curves: dict, filter_prob: float=-1, less_equal=True, param_axis=None, nm_fator_risco='f(.)', show_legend=True, str_title='') -> None:
        """Plot curve

        Args:
            dict_curves (dict): dictionary with all the curves
            filter_prob (float, optional): filter less or greater than some probability (x axis). Defaults to -1.
            less_equal (bool, optional): True if less, False if greater. Defaults to True.
            param_axis (go.Figure, optional): Axis for subplots. Defaults to None.

        """        
        if param_axis is None:
            fig, ax = plt.subplots()
        
        for fator_risco in dict_curves:
            curve = dict_curves[fator_risco]
            
            if filter_prob != -1:
                if less_equal:
                    curve = {key: curve[key] for key in curve.keys() if key <= filter_prob}
                else:
                    curve = {key: curve[key] for key in curve.keys() if key >= filter_prob}
            
            values = curve.values() if type(curve) is dict else curve.values
            
            if self._lib == 'matplotlib':
                if param_axis is None:
                    ax.plot(curve.keys(), values, label=f'{nm_fator_risco}={fator_risco}')
                else:
                    param_axis.plot(curve.keys(), values, label=f'{nm_fator_risco}={fator_risco}')
        
        if show_legend:
            if param_axis is None:
                fig.legend(loc='upper left')
            else:
                param_axis.legend(loc='upper left')
                return param_axis
        
        if param_axis is None:
            fig.xlabel('probability (p\')')
            fig.ylabel('cost (c\')')
            fig.title(str_title)
            return fig, ax
        else:
            param_axis.set_xlabel('probability (p\')')
            param_axis.set_ylabel('cost (c\')')
            param_axis.set_title(str_title)
            return param_axis
                
    def plot_curve_subplots(self, dict_curves: dict, filter_prob: float=-1) -> None:
        """Subplots of curves: (i) full x axis; (ii) less than `filter_prob`; (iii) greater than `filter_prob`

        Args:
            dict_curves (dict): dictionary with all the curves
            filter_prob (float, optional): filter less or greater than some probability (x axis). Defaults to -1.
        """        
        fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(18, 6))
        
        self.plot_curve(dict_curves, -1, True, ax1, show_legend=False)
        self.plot_curve(dict_curves, filter_prob, True, ax2, show_legend=False)
        self.plot_curve(dict_curves, filter_prob, False, ax3, show_legend=False)
        
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels)
        
        plt.show()
        
    def plot_curve_subplots_all_methods(self, _analytical_result: dict, filter_prob: float=-1) -> None:
        """Subplots of curves: (i) full x axis; (ii) less than `filter_prob`; (iii) greater than `filter_prob`

        Args:
            _analytical_result (dict): dictionary with all the curves
            filter_prob (float, optional): filter less or greater than some probability (x axis). Defaults to -1.
        """        
        
        fig, ax = plt.subplots(4, 3, figsize=(18, 16))
        methods = set([m.split('-')[0] for m in _analytical_result.keys()])
        
        for i, method in enumerate(methods):
            dict_curves = {}
            dict_curves[f'{method}-min'] = _analytical_result[f'{method}-min']
            dict_curves[f'{method}-max'] = _analytical_result[f'{method}-max']
            
            self.plot_curve(dict_curves, -1, True, ax[i, 0], show_legend=False)
            self.plot_curve(dict_curves, filter_prob, True, ax[i, 1], show_legend=False)
            self.plot_curve(dict_curves, filter_prob, False, ax[i, 2], show_legend=False)
            
            handles, labels = ax[i, 0].get_legend_handles_labels()
            ax[i, 0].legend(handles, labels) 
            
            ax[i, 0].title.set_text(f'{method} Curve')
            ax[i, 1].title.set_text(f'{method} Curve (p\' < {filter_prob})')
            ax[i, 2].title.set_text(f'{method} Curve (p\' > {filter_prob})')
        
        plt.show()
        
    def plot_all_curves_subplots(self, dict_curves: dict, filter_prob: float=-1) -> None:
        """Subplots of curves: (i) full x axis; (ii) less than `filter_prob`; (iii) greater than `filter_prob`

        Args:
            dict_curves (dict): dictionary with all the curves
            filter_prob (float, optional): filter less or greater than some probability (x axis). Defaults to -1.
        """
        fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(18, 6))
        
        self.plot_curve(dict_curves, -1, True, ax1, show_legend=False, str_title='Comparative Cost Curves: Analytical Curves for \n Each Method in a 2A1S Problem')
        self.plot_curve(dict_curves, filter_prob, True, ax2, show_legend=False, str_title='Comparative Cost Curves: Analytical Curves for \n Each Method in a 2A1S Problem with p\' < 0.5')
        self.plot_curve(dict_curves, filter_prob, False, ax3, show_legend=False, str_title='Comparative Cost Curves: Analytical Curves for \n Each Method in a 2A1S Problem with p\' > 0.5')
        
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels)
        
        plt.show()
        
    def plot_curve_subplots_multiple_probabilities(self, dict_curves: dict, list_p: list) -> None:
        """Multiple probabilities subplots to verify the properties of each curve changing the
        initial probability.

        Args:
            dict_curves (dict): dictionary with all the curves
            list_p (list): list of all probabilities
        """
        if self._lib == 'matplotlib':
            fig, axs = plt.subplots(3, 3, figsize=(16,12))
        elif self._lib == 'plotly':
            fig = go.Figure()
            fig = make_subplots(rows=3, cols=3, width=1600, height=1200)
        
        for risk_factor in dict_curves:
            c1, c2 = 0, 0
            for p in dict_curves[risk_factor]:
                curve = dict_curves[risk_factor][p]
                
                if self._lib == 'matplotlib':
                    axs[c1, c2].plot(curve.keys(), curve.values(), label=f'f()={risk_factor}')
                elif self._lib == 'plotly':
                    fig.add_trace(go.Scatter(
                        x=curve.keys(),
                        y=curve.values(),
                        name=f'f()={risk_factor}',
                        line_color=self._colors[f'f()={risk_factor}'],
                        legendgroup='group1'),
                        row=c1, col=c2
                    )
            
                c1, c2 = self._define_counting(c1, c2, 3)
                
        c1, c2 = 0, 0
        for p in list_p:
            if self._lib == 'matplotlib':
                axs[c1, c2].axvline(x=p, color='green', linestyle='--')
                axs[c1, c2].set_title(f'Comparable Regions for Multiple Probabilities: p = {np.round(p, 1)}')
            # elif self._lib == 'plotly':
                
            c1, c2 = self._define_counting(c1, c2, 3)
            
        if self._lib == 'matplotlib':
            handles, labels = axs[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels)
            
    def plot_all_curves_subplots_multiple_probabilities(self, dict_curves: dict, list_p: list) -> None:
        """Multiple probabilities subplots to verify the properties of each curve changing the
        initial probability.

        Args:
            dict_curves (dict): dictionary with all the curves
            list_p (list): list of all probabilities
        """
        
        if self._lib == 'matplotlib':
            fig, axs = plt.subplots(3, 3, figsize=(16,12))
        elif self._lib == 'plotly':
            titles = []
            for p in list_p:
                titles.append(f'Região ótima - Probabilidade [{np.round(p, 1)}]')
                
            fig = go.Figure() 
            fig = make_subplots(rows=3, cols=3, subplot_titles=titles)
        
        for func in dict_curves:
            for risk_factor in dict_curves[func]:
                c1, c2, show_legend = 0, 0, True
                for p in dict_curves[func][risk_factor]:
                    curve = dict_curves[func][risk_factor][p]
                    risk_factor_label = round(risk_factor, 2) if type(risk_factor) is np.float64 else risk_factor
                    
                    if self._lib == 'matplotlib':
                        axs[c1, c2].plot(curve.keys(), curve.values(), label=f'{func} ({risk_factor_label})')
                    elif self._lib == 'plotly':
                        fig.add_trace(go.Scatter(
                            x=list(curve.keys()),
                            y=list(curve.values()),
                            name=f'{func} ({risk_factor_label})',
                            legendgroup=f'{func} ({risk_factor_label})',
                            line_color=self._colors[f'{func} ({risk_factor_label})'],
                            showlegend=show_legend),
                            row=c1+1, col=c2+1
                        )
                        show_legend=False
                    
                    c1, c2 = self._define_counting(c1, c2, 3)
                
        c1, c2 = 0, 0
        for p in list_p:
            if self._lib == 'matplotlib':
                axs[c1, c2].axvline(x=p, color='green', linestyle='--')
                axs[c1, c2].set_title(f'Região ótima - Probabilidade [{np.round(p, 1)}]')
            elif self._lib == 'plotly':
                fig.add_vrect(x0=p, x1=p, row=c1+1, col=c2+1, opacity=0.5, line_dash="dot")
                
                
            c1, c2 = self._define_counting(c1, c2, 3)
            
        if self._lib == 'matplotlib':
            handles, labels = axs[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels)
        elif self._lib == 'plotly':
            fig.update_layout(
                width=1600, height=1200
            )
            fig.show()
            
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