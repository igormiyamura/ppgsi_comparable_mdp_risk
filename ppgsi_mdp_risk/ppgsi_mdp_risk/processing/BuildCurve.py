
import numpy as np, pandas as pd

class BuildCurve:
    def __init__(self, _cp) -> None:
        self._cp = _cp
    
    def build_curves_empirical(self, c: float, p: float, nm_function: str, _ecc: object, list_neg: list, list_pos: list, _plot=True, **kwargs):
        # Limite Empirico Minimo
        res_neg = {}
        min_neg = np.min(list_neg)

        for rc in list_neg:
            if nm_function == 'ExponentialFunction': params = {'l': rc}
            
            res_neg[rc] = _ecc.get_empirical_limit_curve(
                c=c,
                p=p,
                **params
            )

        df_min = pd.DataFrame(res_neg)
        c_min = df_min[min_neg]

        if _plot:
            self._cp.plot_curve_from_dataframe(
                df=df_min,
                label='p\'',
                transpose=True
            )

        # Limite Empirico max
        res_pos = {}
        max_pos = np.max(list_pos)

        for rc in list_pos:
            if nm_function == 'ExponentialFunction': params = {'l': rc}
            
            res_pos[rc] = _ecc.get_empirical_limit_curve(
                c=c,
                p=p,
                **params
            )

        df_max = pd.DataFrame(res_pos)
        c_max = df_max[max_pos]

        if _plot:
            self._cp.plot_curve_from_dataframe(
                df=df_max,
                label='p\'',
                transpose=True
            )
        
        return res_neg, res_pos, df_min, df_max, c_min, c_max
    
    def build_curves_empirical_multiple_prob(self, c: float, list_p:list, nm_function: str, _ecc: object, list_neg: list, list_pos: list):
        
        dict_curves = {}

        min_max = [min(list_neg), max(list_pos)]

        for rf in min_max:
            if nm_function == 'ExponentialFunction': params = {'l': rf}
            
            dict_curves[rf] = _ecc.get_multi_empirical_limit_curve(c, list_p, **params)

        self._cp.plot_curve_subplots_multiple_probabilities(dict_curves, list_p)
        
        return dict_curves
    
    def build_curves_from_limits(self, c: float, p: float, nm_function: str, _ecc: object):
        # Cria dicionario de curvas
        dict_curves = {}
        limites = ['inf', 'sup']

        # Processa para uma lista de betas
        for lim in limites:
            dict_curves[lim] = _ecc.get_limit_curve(c, p, lim=lim)

        # Cria dataframe com resultados
        df_curves = pd.DataFrame(dict_curves)

        # Realiza plot de cada lim
        self._cp.plot_curve_subplots(
            dict_curves=dict_curves,
            filter_prob=0.5
        )
        
        return df_curves
    
    def build_curves_from_limits_multiple_prob(self, c: float, list_p:list, nm_function: str, _ecc: object):        
        dict_curves = {}
        limites = ['inf', 'sup']

        for lim in limites:
            dict_curves[lim] = _ecc.get_multi_limit_curve(c, list_p, lim=lim)

        self._cp.plot_curve_subplots_multiple_probabilities(dict_curves, list_p)
        
        return dict_curves