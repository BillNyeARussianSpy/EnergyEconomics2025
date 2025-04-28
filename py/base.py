import numpy as np, pandas as pd, pyDbs
from collections.abc import Iterable
from pyDbs import adj, adjMultiIndex, noneInit
from six import string_types
from scipy import sparse
from functools import reduce

def ifinInit(x,kwargs,FallBackVal):
	return kwargs[x] if x in kwargs else FallBackVal

def is_iterable(arg):
	return isinstance(arg, Iterable) and not isinstance(arg, string_types)

def stdNames(k):
	if isinstance(k,str):
		return [f"_{k}symbol", f"_{k}index"]
	else:
		return [n for l in [stdNames(i) for i in k] for n in l]

# GLOBAL INDEX METHODS
def sortAll(v, order=None):
	return reorderStd(v, order=order).sort_index() if isinstance(v, (pd.Series, pd.DataFrame)) else v

def reorderStd(v, order=None):
	return v.reorder_levels(noneInit(order, sorted(pyDbs.getIndex(v).names))) if isinstance(pyDbs.getIndex(v), pd.MultiIndex) else v

def setattrReturn(symbol,k,v):
	symbol.__setattr__(k,v)
	return symbol

def fIndexSeries(variableName, index, btype ='v'):
	return setattrReturn(pd.MultiIndex.from_product([[variableName],reorderStd(index).values], names=stdNames(btype)), '_n', sorted(index.names))

def fIndex(variableName, index, btype = 'v'):
	return setattrReturn(pd.MultiIndex.from_tuples([(variableName,None)], names = stdNames(btype)), '_n', []) if index is None else fIndexSeries(variableName,index, btype = btype)

def fIndexVariable(variableName, v, btype = 'v'):
	return v.set_axis(fIndex(variableName, pyDbs.getIndex(v), btype = btype)) if isinstance(v, pd.Series) else pd.Series(v, index = fIndex(variableName, None, btype=btype), dtype= np.float64)
"""
Changed this:
def vIndexSeries(f,names):
	return f.index.set_names(names) if len(names)==1 else pd.MultiIndex.from_tuples(f.index.values,names=names)

def vIndexSeries(f, names):
    print(f.index.values)  # Debug: Output the index values
    if not all(isinstance(i, tuple) for i in f.index.values):
        raise TypeError("Expected index values to be tuples.")
    return (
        f.index.set_names(names)
        if len(names) == 1
        else pd.MultiIndex.from_tuples(f.index.values, names=names)
    )

def vIndexSeries(f, names):
    if not all(isinstance(i, tuple) for i in f.index.values):
        # Filter out problematic indices
        valid_index = [i for i in f.index.values if isinstance(i, tuple) and not pd.isna(i)]
        if len(valid_index) != len(f.index):
            print(f"Filtered out invalid indices: {set(f.index.values) - set(valid_index)}")
            f = f.loc[valid_index]  # Reindex the Series
    return (
        f.index.set_names(names)
        if len(names) == 1
        else pd.MultiIndex.from_tuples(f.index.values, names=names)
    )

"""
def vIndexSeries(f, names):
    if len(f.index.values) == 0:
        print("Index values are empty; returning empty index.")
        return pd.Index([], name=tuple(names) if isinstance(names, list) else names)
    
    print(f"Index values: {f.index.values}")
    print(f"Names: {names}, Expected levels: {len(f.index.values[0]) if isinstance(f.index.values[0], tuple) else 1}")
    
    if isinstance(f.index.values[0], tuple):
        if len(names) != len(f.index.values[0]):
            print(f"Adjusting names to match levels. Current names: {names}")
            names = names[:len(f.index.values[0])]
        return pd.MultiIndex.from_tuples(f.index.values, names=names)
    else:
        if len(names) != 1:
            print(f"Adjusting names: Expected 1 name, got {len(names)}")
            names = names[:1]  # Trim to one name
    return f.index.set_names(names)

"""
def vIndexVariable(f, variable, names):
    print(f"x-values: {f.xs(variable).values}")
    print(f"Index names: {names}")
    print(f"Generated index: {vIndexSeries(f.xs(variable), names)}")
    return pd.Series(f.xs(variable).values, index = vIndexSeries(f.xs(variable),names), dtype = np.float64) if names else f.xs(variable)[0]
"""
def vIndexVariable(f, variable, names):
    values = f.xs(variable).values
    index = vIndexSeries(f.xs(variable), names)
    if len(values) != len(index):
        # Handle mismatch: provide a fallback index
        print(f"Mismatch: values length = {len(values)}, index length = {len(index)}")
        #Provide hashable index name
        index = pd.Index([None] * len(values), name=tuple(names) if isinstance(names, list) else names)
    return pd.Series(values, index=index, dtype=np.float64)


def vIndexSymbolDual(f, symbol, names):
	keep = f.xs(symbol)
	return keep.set_axis(pd.MultiIndex.from_frame(vIndexSeries(keep.droplevel('_type'), names).to_frame(index=False).assign(_type=keep.index.get_level_values('_type')))) if names else keep.droplevel('_sindex')

# SPARSE METHODS
def sparseSeries(values, index=None, name = None, fill_value = 0, dtype = None):
	""" initialize sparse version of series """
	return pd.Series(pd.arrays.SparseArray(values if is_iterable(values) else np.full(len(index),values), fill_value = fill_value, dtype = dtype), index = index, name = name)

def sparseEmptySeries(size, fill_value=0, index = None, name = None):
	""" initalize sparse version of empty series of given size """
	return pd.Series(pd.arrays.SparseArray(np.empty(size), fill_value=fill_value),index = index, name = name)

def sparseMatrixFromSeries(s, columns):
	colIds = s.groupby(columns).ngroup().values
	rowIds = s.groupby([name for name in s.index.names if name not in columns]).ngroup().values
	return sparse.coo_matrix((s.values, (rowIds, colIds)), shape=(len(rowIds), len(colIds)))

# ITERATION METHODS - not sparse
def sumIte(ite,fill_value=0):
	""" Sum using broadcasting methods on iterative object """
	return reduce(lambda x,y: adjMultiIndex.bcAdd(x,y,fill_value=fill_value), ite)

def maxIte(ite):
	""" Returns max of symbols in ite; ignores NaN unless all columns use it. """
	return pd.concat(ite, axis=1).max(axis=1) if isinstance(ite[0], pd.Series) else max([noneInit(x,np.nan) for x in ite])

def minIte(ite):
	return pd.concat(ite, axis=1).min(axis=1) if isinstance(ite[0], pd.Series) else min([noneInit(x,np.nan) for x in ite])

def stackValues(ite):
	return np.hstack([f.values for f in ite])

def stackIndex(ite,names):
	return pd.MultiIndex.from_tuples(np.hstack([pyDbs.getIndex(f).values for f in ite]), names = names)

def stackSeries(ite, names):
	return pd.Series(stackValues(ite), index = stackIndex(ite,names), dtype = np.float64)

def stackIte(varsDict,fill_value=0, btype = 'v', addFullIndex= True):
	""" Returns stacked variable with global index"""
	fvars = [fIndexVariable(k, varsDict[k]) for k in varsDict] if addFullIndex else varsDict.values()
	return stackSeries(fvars, names = stdNames(btype))

# Auxiliary functions that help create suitable parameter inputs:
def appIndexWithCopySeries(s, copyLevel, newLevel):
	s.index = appendIndexWithCopy(s.index,copyLevel,newLevel)
	return s

def appendIndexWithCopy(index, copyLevel, newLevel):
	if is_iterable(copyLevel):
		return pd.MultiIndex.from_frame(index.to_frame(index=False).assign(**{newLevel[i]: index.get_level_values(copyLevel[i]) for i in range(len(copyLevel))}))
	else: 
		return pd.MultiIndex.from_frame(index.to_frame(index=False).assign(**{newLevel: index.get_level_values(copyLevel)}))

def rollLevelS(s, level, offset, order = None):
	s.index = rollLevel(s.index, level, offset, order = order)
	return s

def rollLevel(index, level, offset, order = None):
	if is_iterable(level):
		if order is not None:
			return index.set_levels([index.levels[index.names.index(level[i])].map(_offsetMap(order[i], offset[i])) for i in range(len(level))], level = level)
		else:
			return index.set_levels([np.roll(index.levels[index.names.index(level[i])],offset[i]) for i in range(len(level))], level = level)
	else:
		if order is not None:
			return index.set_levels(index.levels[index.names.index(level)].map(_offsetMap(order, offset)), level = level)
		else:
			return index.set_levels(np.roll(index.levels[index.names.index(level)], offset), level = level)

def offsetLevelS(s, level, offset):
	s.index = offsetLevel(s.index, level, offset)
	return s

def offsetLevel(index, level, offset):
	if is_iterable(level):
		return index.set_levels([index.levels[index.names.index(level[i])]+offset[i] for i in range(len(level))], level = level)
	else:
		return index.set_levels(index.levels[index.names.index(level)]+offset, level = level)

def _offsetMap(index, offset):
	return pd.Series(np.roll(index, offset), index = index)

def pdNonZero(x):
	return x.where(x!=0)
