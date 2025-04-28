import numpy as np, pandas as pd
from scipy import optimize, stats

import pandas as pd
import scipy.optimize

class MAC:
    def __init__(self, α=0.5, γ=1, pe=1, ϕ=0.25, γd=100):
        # Storing parameters in a dictionary
        self.db = {'α': α, 'γ': γ, 'pe': pe, 'ϕ': ϕ, 'γd': γd}
    
    def C(self, Egrid):
        # Access parameters from the db
        α = self.db['α']
        γ = self.db['γ']
        pe = self.db['pe']
        return pd.Series(γ * Egrid**α - pe * Egrid, name='C', index=Egrid)

    def M(self, Egrid):
        # Access parameters from the db
        ϕ = self.db['ϕ']
        return pd.Series(ϕ * Egrid, name='M', index=Egrid)
    
    def MAC(self, E):
        # Access parameters from the db
        α = self.db['α']
        γ = self.db['γ']
        pe = self.db['pe']
        ϕ = self.db['ϕ']
        return (γ * α * E**(α-1) - pe) / ϕ
    
    def Ctilde(self, E):
        # Use C and D methods
        return self.C(E) - self.D(self.M(E))
    
    def D(self, M):
        # Access parameter from the db
        γd = self.db['γd']
        return (γd * M**2) / 2
    
    def E0(self):
        # Access parameters from the db
        α = self.db['α']
        γ = self.db['γ']
        pe = self.db['pe']
        return ((γ * α) / pe) ** (1 / (1 - α))
    
    def C0(self, E):
        # Access parameters from the db
        α = self.db['α']
        γ = self.db['γ']
        pe = self.db['pe']
        return γ * E**α - pe * E
    
    def M0(self, E):
        # Access parameter from the db
        ϕ = self.db['ϕ']
        return ϕ * E
    
    def Eopt(self):
        # Access parameters from the db
        α = self.db['α']
        γ = self.db['γ']
        pe = self.db['pe']
        ϕ = self.db['ϕ']
        γd = self.db['γd']
        # Optimization to find Eopt
        return scipy.optimize.fsolve(lambda E: α * γ * E**(α-1) - pe - γd * ϕ**2 * E, 0.5)[0]
    
    def Copt(self):
        # Use Eopt to calculate Copt
        α = self.db['α']
        γ = self.db['γ']
        pe = self.db['pe']
        Eopt = self.Eopt()
        return γ * Eopt**α - pe * Eopt
    
    def Mopt(self):
        # Use Eopt
        ϕ = self.db['ϕ']
        Eopt = self.Eopt()
        return ϕ * Eopt

class MACTech(MAC):
	def __init__(self, α = .5, γ = 1, pe = 1, ϕ = .25, γd = 100, θ = None, c = None, σ = None):
		super().__init__(α = α, γ = γ, pe = pe, ϕ = ϕ, γd = γd) # use __init__ method from parent class
		self.initTechs(θ = θ, c = c, σ = σ)
		
	def initTechs(self, θ = None, c = None, σ = None):
		""" Initialize technologies from default values """
		if θ is None:
			self.db['Tech'] = pd.Index(['T1'], name = 'i')
			self.db['θ'] = pd.Series(0, index = self.db['Tech'], name = 'θ')
			self.db['c'] = pd.Series(1, index = self.db['Tech'], name = 'c')
			self.db['σ'] = pd.Series(1, index = self.db['Tech'], name = 'σ')
		elif isinstance(θ, pd.Series):
			self.db['θ'] = θ
			self.db['c'] = c
			self.db['σ'] = σ
			self.db['Tech'] = self.db['θ'].index
		else:
			self.db['Tech'] = 'T'+pd.Index(range(1, len(θ)+1), name = 'i').astype(str)
			self.db['θ'] = pd.Series(θ, index = self.db['Tech'], name = 'θ')
			self.db['c'] = pd.Series(c, index = self.db['Tech'], name = 'c')
			self.db['σ'] = pd.Series(σ, index = self.db['Tech'], name = 'σ')
