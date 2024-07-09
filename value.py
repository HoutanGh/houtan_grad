class Value:
    def __init__(self, data, _children=(), _op=''):
        # Want to keep track of all the data of values that make current data, so introduce _children

        self.data = data
        self._prev = set(_children) # This is for efficiency, removing duplicates
        self._op = _op
    
    def __repr__(self) -> str:
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out
    

    
    
    