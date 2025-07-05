from interval import *

class ComplexInterval:
    def __init__(self, real: interval, imag: interval):
        self.real = real
        self.imag = imag

    def __repr__(self):
        real = [round(el, 3)for el in list(self.real[0])]
        imag = [round(el, 3)for el in list(self.imag[0])]
        return f"<{real}, {imag}>"
    
    def __add__(self, other):
        return ComplexInterval(self.real + other.real, self.imag + other.imag)
    
    def __sub__(self, other):
        return ComplexInterval(self.real - other.real, self.imag - other.imag)
    
    def __mul__(self, other):
        return ComplexInterval(self.real*other.real - self.imag*other.imag, self.real*other.imag + self.imag*other.real)
    
    def __rmul__(self, other):
        # assert type(other) == complex, f"Expected 'complex', got {type(other)}"

        return self.__mul__(ComplexInterval(interval([other.real, other.real]), interval([other.imag, other.imag])))
    
    def abs_powered(self):
        return self.real**2 + self.imag**2

    def clip(self):
        self.real = self.real & interval([-1, 1])
        self.imag = self.imag & interval([-1, 1])


