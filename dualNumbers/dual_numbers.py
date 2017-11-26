import numba
import numpy as np
from numba import vectorize, float64, int32
from prettytable import PrettyTable

# ----- Define numba optimized functions
@vectorize([float64(float64, float64), int32(int32, int32)], target="parallel")
def vec_sum(x, y):
    return x + y

@vectorize([float64(float64, float64), int32(int32, int32)], target="parallel")
def vec_sub(x, y):
    return x - y

@vectorize([float64(float64, float64), int32(int32, int32)], target="parallel")
def vec_mul(x, y):
    return x * y

@vectorize([float64(float64, float64), int32(int32, int32)], target="parallel")
def vec_truediv(x, y):
    return x / y

@vectorize([float64(float64, float64), int32(int32, int32)], target="parallel")
def vec_pow(x, y):
    return x**y

@vectorize([float64(float64, float64), int32(int32, int32)], target="parallel")
def vec_one(x, y):
    return 1

@vectorize([float64(float64, float64), int32(int32, int32)], target="parallel")
def vec_mul_deriv_x(x, y):
    return y

@vectorize([float64(float64, float64), int32(int32, int32)], target="parallel")
def vec_mul_deriv_y(x, y):
    return x

@vectorize([float64(float64, float64), int32(int32, int32)], target="parallel")
def vec_div_deriv_x(x, y):
    return 1/y

@vectorize([float64(float64, float64), int32(int32, int32)], target="parallel")
def vec_div_deriv_y(x, y):
    return -x/y**2

@vectorize([float64(float64, float64), int32(int32, int32)], target="parallel")
def vec_pow_deriv_x(x, y):
    return y*x**(y-1)

@vectorize([float64(float64, float64), int32(int32, int32)], target="parallel")
def vec_pow_deriv_y(x, y):
    return x**y*np.log(x)

@vectorize([float64(float64), int32(int32)], target="parallel")
def vec_exp(x):
    return np.exp(x)

@vectorize([float64(float64), int32(int32)], target="parallel")
def vec_log(x):
    return np.log(x)

@vectorize([float64(float64), int32(int32)], target="parallel")
def vec_log_deriv(x):
    return 1/x

@vectorize([float64(float64), int32(int32)], target="parallel")
def vec_sqrt(x):
    return np.sqrt(x)


# ----- Define Dual class 
class Dual(object):

    # Maximum number of lines to print
    max_print_lines = 100

    def __init__(self, real, **kwargs):
        # Real can be initilized as np array or int/float
        if isinstance(real, (int, float)):
            self.real = np.array([real])
        else:
            self.real = real

        # Initilization of duals
        self.duals = list() # Preserve the order arguments were given in
        for key, val in kwargs.items(): # As kwargs is unordered (For <python3.6) items in the loop appear in random order
            if isinstance(val, int):
                if val == 1:
                    setattr(self, key , np.ones(self.real.shape[0]))
                else:
                    raise ValueError("Wrong initilization of dual part")
            else:
                setattr(self, key , val)
            self.duals.append(key)

    def func_template(self, func, func_deriv):
        val = { key: func_deriv(self.real) * vars(self)[key] for key in set(self.duals)}
        real = func(self.real)
        return Dual(real, **val)

    def func_template_2d(self, other, func, func_deriv_x, func_deriv_y):
        if isinstance(other, Dual):
            # The following can be considered as taking the sets X and Y, and splitting into three cases: X\Y, Y\X and X&Y

            # X\Y
            val = { key: func_deriv_x(self.real, other.real) * vars(self)[key] for key in set(self.duals) - set(other.duals)}

            # Y\X
            val.update({ key: func_deriv_y(self.real, other.real) * vars(other)[key] for key in set(other.duals) - set(self.duals)})

            # X&Y
            val.update({ key: (func_deriv_x(self.real, other.real) * vars(self)[key] + func_deriv_y(self.real, other.real) * vars(other)[key]) for key in set(self.duals) & set(other.duals)})

            # Function of real parts
            real = func(self.real, other.real)
            
        elif isinstance(other, (int,float, np.ndarray)):
            # If second argument is a int or float, the y-derivative term becomes zero.
            val = { key: func_deriv_x(self.real, other) * vars(self)[key] for key in set(self.duals)}
            real = func(self.real, other)
        else:
            raise ValueError("Type not supported")
        
        return Dual(real, **val)
    
    def __add__(self, other):
        return self.func_template_2d(other, vec_sum, vec_one, vec_one)

    def __sub__(self, other):
        return self.__add__((other*(-1)))

    def __mul__(self, other):
        return self.func_template_2d(other, vec_mul, vec_mul_deriv_x, vec_mul_deriv_y)

    def __truediv__(self, other):
        return self.func_template_2d(other, vec_truediv, vec_div_deriv_x, vec_div_deriv_y)
    
    def __pow__(self, other):
        return self.func_template_2d(other, vec_pow, vec_pow_deriv_x, vec_pow_deriv_y)
    
    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return self.negate().__add__(other)
        else:
            raise ValueError("Type not supported")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def exp(self):
        # The derivative is equal to the value of the function itself, when considering an exponential function.
        return self.func_template(vec_exp, vec_exp)

    def log(self):
        return self.func_template(vec_log, vec_log_deriv)
    
    def sin(self):
        return self.func_template(np.sin, np.cos)

    def cos(self):
        return self.func_template(np.cos, (-1)*np.sin)

    def sqrt(self):
        return self.func_template(vec_sqrt, 1/(2*vec_sqrt))
    
    def __neg__(self):
        return self.negate()
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        # Initilize prettytable
        table = PrettyTable(['Real'] + self.duals)
        table.aling = 'r'
        # Create list of dual vectors
        dual_list = [vars(self)[key] for key in self.duals]
        
        # Create Numpy matrix for printing
        data = np.asarray(np.matrix([self.real] + dual_list).T)

        # Rows to print
        n_lines = min(Dual.max_print_lines, self.real.shape[0])
        
        # Add rows to prettytable
        for count, row in enumerate(data):
            if count == n_lines:
                break
            else:
                table.add_row(row)
        return table.get_string()

    def negate_duals(self):
        duals = { key: vec_mul((-1), vars(self)[key]) for key in set(self.duals) }
        return duals

    def negate(self):
        duals = self.negate_duals()
        real = vec_mul((-1), self.real)
        return Dual(real, **duals)

if __name__ == "__main__":

    aa = Dual(1, x=1)
    bb = Dual(10, y=1)
    
    # Test functions
    n = 10**3
    x = np.random.normal(0, 1, n)
    y = np.random.normal(0, 1, n)
    z = np.random.exponential(1, n)
    v = np.random.exponential(0.3, n)

    d1 = Dual(x, x=1)
    d2 = Dual(y, y=1)
    d3 = Dual(z, z=1)
    d4 = Dual(v, v=1)

    # Addition
    add = lambda x, y: x + y
    add_x = lambda x, y: 1
    add_y = lambda x, y: 1

    assert np.isclose(add(d1,d2).real, add(x,y)).all()
    assert np.isclose(add(d1,d2).x, add_x(x,y)).all()
    assert np.isclose(add(d1,d2).y, add_y(x,y)).all()

    assert np.isclose((d1 + x).real, x + x).all()
    
    # Subtraction
    sub = lambda x, y: x - y
    sub_x = lambda x, y: 1
    sub_y = lambda x, y: -1

    assert np.isclose(sub(d1,d2).real, sub(x,y)).all()
    assert np.isclose(sub(d1,d2).x, sub_x(x,y)).all()
    assert np.isclose(sub(d1,d2).y, sub_y(x,y)).all()

    # Multiplication
    mul = lambda x, y: x*y
    mul_x = lambda x, y: y
    mul_y = lambda x, y: x

    assert np.isclose(mul(d1,d2).real, mul(x,y)).all()
    assert np.isclose(mul(d1,d2).x, mul_x(x,y)).all()
    assert np.isclose(mul(d1,d2).y, mul_y(x,y)).all()

    # Division
    div = lambda x, y: x/y
    div_x = lambda x, y: 1/y
    div_y = lambda x, y: (-1)*x/(y*y)

    assert np.isclose(div(d1,d2).real, x/y).all()
    assert np.isclose(div(d1,d2).x, div_x(x,y)).all()
    assert np.isclose(div(d1,d2).y, div_y(x,y)).all()

    # Power
    power = lambda z, v: z**v
    power_z = lambda z, v: v*z**(v-1)
    power_v = lambda z, v: z**v*np.log(z)

    assert np.isclose(power(d3,d4).real, power(z,v)).all()
    assert np.isclose(power(d3,d4).z, power_z(z,v)).all()
    assert np.isclose(power(d3,d4).v, power_v(z,v)).all()

    assert np.isclose(power(d1,2).real, power(x,2)).all()
    assert np.isclose(power(d1,2).x, power_z(x,2)).all()

    power2 = lambda z, v: 1.1-z**2*1/2*v + z*v
    power2_z = lambda z, v: -z*v + v
    power2_v = lambda z, v: -z**2*1/2 + z

    assert np.isclose(power2(d3,d4).real, power2(z,v)).all()
    assert np.isclose(power2(d3,d4).z, power2_z(z,v)).all()
    assert np.isclose(power2(d3,d4).v, power2_v(z,v)).all()

    
    # Exp
    assert np.isclose(d1.exp().real, np.exp(x)).all()
    assert np.isclose(d1.exp().x, np.exp(x)).all()

    # Larger function
    f = lambda x, y: -x + x*x+2*y + 2 + x*y + x/y 
    f_x = lambda x, y: -1+2*x+y+1/y
    f_y = lambda x, y: 2 + x -x/y**2
    assert np.isclose(f(d1,d2).real, f(x,y)).all()
    assert np.isclose(f(d1,d2).x, f_x(x,y)).all()
    assert np.isclose(f(d1,d2).y, f_y(x,y)).all()
