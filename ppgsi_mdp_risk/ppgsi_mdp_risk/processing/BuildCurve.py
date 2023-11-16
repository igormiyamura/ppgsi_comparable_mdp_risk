
import numpy as np, pandas as pd

class BuildCurve:
    def __init__(self, _cp) -> None:
        self._cp = _cp
    
    def build_curves_empirical_from_list(self, c: float, p: float, nm_function: str, _ecc: object, list_rf: list, _plot=True, _obj_function: object=None, **kwargs):
        # Limite Empirico Minimo
        res = {}

        for rf in list_rf:
            params = self._get_params(nm_function, rf, _obj_function)
            
            res[rf] = _ecc.get_empirical_limit_curve(
                c=c,
                p=p,
                **params,
                **kwargs
            )

        df = pd.DataFrame(res)

        if _plot:
            self._cp.plot_curve_subplots(
                dict_curves=res,
                filter_prob=0.5
            )
            
        return res, df
    
    def build_curves_empirical(self, c: float, p: float, nm_function: str, _ecc: object, list_neg: list, list_pos: list, _plot=True, _obj_function: object=None, **kwargs):
        # Limite Empirico Minimo
        res_neg = {}
        min_neg = np.min(list_neg)

        for rf in list_neg:
            params = self._get_params(nm_function, rf, _obj_function)
            
            res_neg[rf] = _ecc.get_empirical_limit_curve(
                c=c,
                p=p,
                **params,
                **kwargs
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

        for rf in list_pos:
            params = self._get_params(nm_function, rf, _obj_function)
            
            res_pos[rf] = _ecc.get_empirical_limit_curve(
                c=c,
                p=p,
                **params,
                **kwargs
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
    
    def build_curves_empirical_multiple_prob(self, c: float, list_p:list, nm_function: str, _ecc: object, list_neg: list, list_pos: list, _plot=True, _obj_function: object=None, **kwargs):
        
        dict_curves = {}

        min_max = [min(list_neg), max(list_pos)]

        for rf in min_max:
            params = self._get_params(nm_function, rf, _obj_function)
            
            dict_curves[rf] = _ecc.get_multi_empirical_limit_curve(c, list_p, **params, **kwargs)

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
    
    def build_empirical_graph_limits_each_function(self, _empirical_result):        
        self._cp.plot_all_curves_subplots(_empirical_result, filter_prob=0.5)
        
        _empirical_result_min = {}
        _empirical_result_max = {}
        
        for _func in _empirical_result:
            if 'min' in _func:
                _empirical_result_min[_func] = _empirical_result[_func]
            elif 'max' in _func:
                _empirical_result_max[_func] = _empirical_result[_func]
                
        self._cp.plot_all_curves_subplots(_empirical_result_min, filter_prob=0.5)
        self._cp.plot_all_curves_subplots(_empirical_result_max, filter_prob=0.5)
    
    def _get_params(self, nm_function, rf, _obj_function, **kwargs):
        if nm_function == 'ExponentialFunction': 
            params = {'l': rf}
        elif nm_function == 'PiecewiseTransformation': 
            alpha, gamma = _obj_function._get_alpha(rf), 1
            params = {'k': rf, 'alpha': alpha, 'gamma': gamma}
        elif nm_function == 'PolynomialFunction': 
            params = {'beta': rf}
        elif nm_function == 'VAR': 
            params = {'alpha': rf}
        else:
            raise Exception(f'[_get_params]: Funcao não definida: [{nm_function}]')
        
        return params