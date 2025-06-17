from astropy.time import Time
from typing import Dict, Any, Optional
import numpy as np

# Define a custom class for observation night outside of methods
class obsNightParams:
    """Class to store observation night information."""
    
    def __init__(self):
        # Initialize attributes that will be set later
        self.sunrise_civil: Optional[Time] = None
        self.sunset_civil: Optional[Time] = None
        self.sunrise_nautical: Optional[Time] = None
        self.sunset_nautical: Optional[Time] = None
        self.sunrise_astro: Optional[Time] = None
        self.sunset_astro: Optional[Time] = None
        self.sunrise_night: Optional[Time] = None
        self.sunset_night: Optional[Time] = None

    def __getattr__(self, name: str):
        # Only return None for expected time attributes
        expected_attrs = ['sunrise_civil', 'sunset_civil', 'sunrise_nautical', 'sunset_nautical', 
                        'sunrise_astro', 'sunset_astro', 'sunrise_night', 'sunset_night']
        if name in expected_attrs:
            return None
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __repr__(self) -> str:
        def format_value(value):
            # Handle lists and other iterables
            if isinstance(value, (list, np.ndarray)):
                if len(value) > 5:
                    return f'[{len(value)} items]'
                return str(value)
            return str(value)

        attrs = {name: format_value(value) for name, value in self.__dict__.items()}
        max_key_len = max(len(str(key)) for key in attrs.keys()) if attrs else 0
        attrs_str = '\n'.join([f'{key:{max_key_len}}: {value}' for key, value in attrs.items()])
        return f'Night Attributes:\n{attrs_str}'

class staraltParams:
    def __init__(self, data_dict=None):
        # Store the original dictionary
        self._data_dict = data_dict or {}
        
        # Convert dictionary keys to class attributes
        if data_dict:
            for key, value in data_dict.items():
                setattr(self, key, value)
    
    @property
    def data_dict(self) -> Dict[str, Any]:
        return self._data_dict
    
    def __getattr__(self, name):
        return None

    def __getitem__(self, key):
        return getattr(self, key, None)
    
    def __contains__(self, key):
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __repr__(self):
        def format_value(value):
            # Handle lists and other iterables
            if isinstance(value, (list, np.ndarray)):
                if len(value) > 5:
                    return f'[{len(value)} items]'
                return str(value)
            return str(value)

        attrs = {name: format_value(value) for name, value in self.__dict__.items() if not name.startswith('_')}
        max_key_len = max(len(str(key)) for key in attrs.keys()) if attrs else 0
        attrs_str = '\n'.join([f'{key:{max_key_len}}: {value}' for key, value in attrs.items()])
        return f'Data Attributes:\n{attrs_str}'
