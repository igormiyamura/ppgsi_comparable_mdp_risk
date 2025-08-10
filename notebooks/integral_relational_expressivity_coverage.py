
import numpy as np
from scipy.integrate import quad

# INTEGRAL - RELATIONAL EXPRESSIVITY COVERAGE

if __name__ == '__main__':
    f_lower_bound = lambda c: c
    f_upper_bound = lambda c, p, p_: c * np.log(1 - p_) / np.log(1 - p)

    def integrand(x, c, p_):
        return np.abs(f_upper_bound(c, x, p_) - f_lower_bound(c))
    
    c, p_ = 1, 0.5
    eum_integral = quad(integrand, 0.01, 0.99, args=(c, p_))
    print(f'Integral for c = {c} and p\' = {p_} is {eum_integral}.')
    
    # =================================================================
    f_lower_bound = lambda c: c
    f_upper_bound = lambda c, p, p_: c * (p_ * (1 - p)) / (p * (1 - p_))

    def integrand(x, c, p_):
        return np.abs(f_upper_bound(c, x, p_) - f_lower_bound(c))
    
    c, p_ = 1, 0.5
    eum_integral = quad(integrand, 0.01, 0.99, args=(c, p_))
    print(f'Integral for c = {c} and p\' = {p_} is {eum_integral}.')
    
    # =================================================================
    f_lower_bound = lambda c: c
    f_upper_bound = lambda c, p, p_: c * (p_ * np.sqrt(1 - p)) / (p * (1 - p_))

    def integrand(x, c, p_):
        return np.abs(f_upper_bound(c, x, p_) - f_lower_bound(c))
    
    c, p_ = 1, 0.5
    eum_integral = quad(integrand, 0.01, 0.99, args=(c, p_))
    print(f'Integral for c = {c} and p\' = {p_} is {eum_integral}.')