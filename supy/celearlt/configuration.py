import json
import yaml
from pathlib import Path
from astropy.time import Time
import numpy as np
from typing import Union, Any, Dict

class Configuration:
    """
    Base Configuration class for all modules.
    Loads multiple YAML (or JSON) config files and merges them.
    Later files override earlier ones.

    - Attribute-style access: config.foo -> self._dict["foo"]
    - Is picklable (can be sent to multiprocessing workers)
    """

    def __init__(self, config_filenames: Union[str, list[str]]):
        # use __dict__ directly here to avoid __setattr__ logic during init
        self.__dict__["_dict"] = {}
        self.__dict__["_config_paths"] = []

        config_filenames = np.atleast_1d(config_filenames)

        for filename in config_filenames:
            config_path = Path(__file__).parent / filename
            try:
                self._dict.update(self._load_config(config_path))
                self._config_paths.append(config_path)
            except FileNotFoundError:
                print(f"⚠️ Config file {config_path} not found, skipping.")

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self):
        # Convert Time objects nicely for printing
        attrs: Dict[str, Any] = {
            k: (v.iso if isinstance(v, Time) else v)
            for k, v in self._dict.items()
        }
        max_key_len = max((len(key) for key in attrs.keys()), default=0)
        attrs_str = "\n".join(
            [f"{k:{max_key_len}} : {v}" for k, v in attrs.items()]
        )
        return (f"===== Configuration =====\n"
                f"{attrs_str}")

    # ------------------------------------------------------------------
    # Attribute access
    # ------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        """
        Called only if normal attribute lookup fails.
        We redirect public attribute access to the internal dict.
        """
        # avoid touching private attributes here, prevent recursion
        if name.startswith("_"):
            raise AttributeError(f"Attribute {name} not found")

        d = self.__dict__.get("_dict", {})
        if name in d:
            return d[name]
        raise AttributeError(f"Attribute {name} not found")

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Public attributes go into the config dict.
        Private attributes (starting with "_") go to the instance dict.
        """
        if name.startswith("_"):
            # bypass our own logic for private attrs
            super().__setattr__(name, value)
        else:
            # store user-facing config values in _dict
            self._dict[name] = value

    # ------------------------------------------------------------------
    # Pickling support
    # ------------------------------------------------------------------
    def __getstate__(self):
        """
        Return a picklable representation of the object.
        Only store the essential data, not methods or file handles.
        """
        return {
            "_dict": self._dict,
            "_config_paths": [str(p) for p in self._config_paths],
        }

    def __setstate__(self, state):
        """
        Restore object from pickled state.
        """
        self.__dict__["_dict"] = state["_dict"]
        self.__dict__["_config_paths"] = [Path(p) for p in state["_config_paths"]]

    # ------------------------------------------------------------------
    # Loading & helpers
    # ------------------------------------------------------------------
    def _load_config(self, config_path: Path) -> dict:
        suffix = config_path.suffix.lower()

        with open(config_path, "r") as f:
            if suffix in [".yaml", ".yml", ".config"]:
                return yaml.safe_load(f) or {}
            elif suffix == ".json":
                return json.load(f) or {}
            else:
                raise ValueError(f"❌ Unsupported config format: {suffix}")

    def update(self, **kwargs) -> None:
        """Update config dictionary with provided key-value pairs."""
        self._dict.update(kwargs)

    def to_plain_dict(self) -> dict:
        """
        Return a plain Python dict of the configuration contents.
        This is often better to send to multiprocessing workers than the
        full Configuration object itself.
        """
        # Note: if you have non-JSON-serializable objects in _dict,
        # you can customize this behavior.
        def default(o):
            if isinstance(o, Time):
                # store as ISO string or MJD if you want
                return o.iso
            return str(o)

        # round-trip through JSON for a deep copy of only serializable content
        return json.loads(json.dumps(self._dict, default=default))
