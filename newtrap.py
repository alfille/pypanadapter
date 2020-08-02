#!/bin/python3

# Newton-Raphon method process control
# The idea is to find the input 'x' that gives an output closest to target
#
# There is (optional) bounding limits on x: 'lo' and 'hi' 
# you can start with an (optional) initial guess 'x0'
# You can set the allowable error 'error' margin (2-sided)
#
# needs no other modules
#
# Method:
# Newton-Raphson for finding a function zero:
# x1 = x0 - f(x0)/f'(x0)
#
# since we don't have a derivative, we need to points to get a difference
# We could choose 2 close points, but that would lead to instability with small output errors
# so we choose the midpoint
#
# Written by Paul H Alfille 2020
# MIT license
#
# see https://github.com/alfille/NewtRap
#
# Usage:
# import NewtRap
# target = 10
# error = .2
# x0 = 1
# lo = 0
# hi = 2
# nr = NewtRap.NewtRap( target, error=error, lo=lo, hi=hi, x0=x0 )
#
# x = x0
# while True:
#     y = my_process(x)
#     print(x,y)
#     x = nr.next(y)

class NewtRap():
    # Newton Raphson 's method for control
    # Uses 2 points to find derivative, so needs 2 measurements
    # Remembers internally which measurement
    # Lots of care with all the special cases
    def __init__(self, target=1, error = None, lo=None, hi=None, x0=None):
        self._target = target
        
        if error is not None:
            self._error = abs(error)
        elif self._target == 0:
            self._error = .01
        else:
            self._error = .01 * abs(self._target)

        # sort and set lo and hi
        self._lo = lo
        self._hi = hi        
        if lo is not None and hi is not None:
            if lo > hi:
                s = hi
                hi = lo
                lo = s
            elif lo == hi:
                hi = lo + 1 # arbitrary
            self._lo = lo
            self._hi = hi
 
        # initial x's -- lot's of cases
        if x0 is not None:
            self.xpair0, self.xpair1 = ( x0, x0+1 )
        elif lo is None: # no lo        
            if hi is None:
                self.xpair0, self.xpair1 = ( .5, 1.5 )
            else:
                self.xpair0, self.xpair1 = ( hi, hi-1 )
        elif hi is None: # no lo
            if lo is None:
                self.xpair0, self.xpair1 = ( .5, 1.5 )
            else:
                self.xpair0, self.xpair1 = ( lo, lo+1 )
        else: # bounded
            self.xpair0, self.xpair1 = (lo,hi)
        
        self.new_settings()
        
    def next( self, value ):

        # prime the pump
        if self.very_first:
            # ignore value (no context)
            self.very_first = False
            self.first = True # which part of the pair?
            return self.xpair0
            
        # value is from previous x
        y = value - self._target

        if self.first:
            # from xpair0
            self.ypair0 = y
            if abs(y) <= self._error:
                # within tolerances, repeat
                return self.xpair0
            else:
                self.first = False
                return self.xpair1
        else:
            # from xpair1
            self.ypair1 = y
            if abs(y) <= self._error:
                # within tolerances, repeat
                return self.xpair1
            else:
                self.xpair0, self.xpair1  = self.new_pair()
                self.apply_limits()
                self.first = True
                return self.xpair0

    def new_pair( self ):
        # average and difference
        x1 = .5 * ( self.xpair0 + self.xpair1 )
        y1 = .5 * ( self.ypair0 + self.ypair1 )
        dx = self.xpair0 - self.xpair1
        dy = self.ypair0 - self.ypair1
        
        if dx == 0:
            #print("X match")
            return self.adjust()
        
        # "minimum"
        if dy == 0 :
            #print("Y match")
            # move a little and remeasure
            return self.adjust()

        # method
        x2 = x1 - y1 * dx / dy
        
        # New bracket
        return ( x2, .5 * (x1 + x2) )

    def adjust( self ):
        # called when calculation is unstable
        # Jostle a bit and remeasure
        return ( self.xpair0, .5* (self.xpair0+self.xpair1) )
        
    def apply_limits( self ):
        if self.xpair0 == self.xpair1:
            self.xpair1 = self.xpair0 + 1
        if self._lo is not None:
            if self.xpair0 < self._lo:
                self.xpair0 = self._lo
            if self.xpair1 < self._lo:
                self.xpair1 = self._lo
            if self.xpair0 == self.xpair1:
                self.xpair1 = self.xpair0+1
        if self._hi is not None:
            if self.xpair0 > self._hi:
                self.xpair0 = self._hi
            if self.xpair1 > self._hi:
                self.xpair1 = self._hi
            if self.xpair0 == self.xpair1:
                self.xpair1 = self.xpair0-1
                    
    @property
    def target( self ):
        return self._target
        
    @target.setter
    def target( self, t ):
        # Big jostle
        self.xpair0 += t - self._target
        # New target
        self._target = t
        self.new_settings()

    @property
    def error( self ):
        return self._error
        
    @error.setter
    def error( self, e ):
        self._error = e
        self.new_settings()

    @property
    def lo( self ):
        return self._lo
        
    @lo.setter
    def lo( self, e ):
        self._lo = e
        self.new_settings()

    @property
    def hi( self ):
        return self._lo
        
    @hi.setter
    def hi( self, e ):
        self._hi = e
        self.new_settings()

    def new_settings( self ):
        # for any change in parameters
        self.very_first = True
        self.apply_limits()
        
