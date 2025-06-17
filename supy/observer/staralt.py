from datetime import datetime as dt, timedelta
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt

from typing import Union, Dict, Any, Optional, Tuple

from matplotlib.dates import date2num, DateFormatter
from datetime import datetime
import logging
from .mainobserver import mainObserver
from .params import obsNightParams, staraltParams
from . import bumper

class Staralt():
    """Class to handle star altitude plots."""
    
    def __init__(self, 
                observer: Optional[mainObserver] = None,
                utctime: Optional[Union[dt, Time]] = None,
                logger: Optional[logging.Logger] = None):
        """
        Initialize a Staralt object.
        
        Parameters
        ----------
        observer : mainObserver
            Observer object with location information.
        utctime : datetime or Time, optional
            Initial time for calculations. Default is current time.
        logger : logging.Logger, optional
            Logger instance for debugging. If None, creates a basic logger.
        """
        # Initialize logger first - CRITICAL FIX
        if logger is not None:
            self.logger = logger
            self.logger.debug("Staralt initialized with external logger")
        else:
            # Create a basic logger if none provided
            self.logger = logging.getLogger(f"{__name__}.Staralt")
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - [Staralt] - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.WARNING)  # Set to WARNING to avoid spam
                self.logger.propagate = False
            self.logger.debug("Staralt initialized with default logger")
        
        # Rest of the original __init__ code...
        if observer is None:
            observer = mainObserver()
        
        self._observer = observer
        
        if utctime is None:
            utctime = self._observer.now()
        if not isinstance(utctime, Time):
            utctime = Time(utctime)

        self._set_night(utctime)
            
    @property
    def utctime(self) -> Time:
        return self._observer.now()
    
    @property
    def observer(self) -> mainObserver:
        """Get the observer object."""
        return self._observer
    
    @property
    def target_coord(self) -> SkyCoord:
        return self._target_coord
    
    @property
    def objname(self) -> str:
        return self._objname
    
    @property
    def target_altaz(self) -> AltAz:
        if not hasattr(self, '_target_coord') or self._target_coord is None:
            raise ValueError("Target coordinates not set. Call set_target() first.")
        if not hasattr(self.observer, '_earthlocation'):
            raise AttributeError("Observer location not properly initialized.")
        return self._target_coord.transform_to(AltAz(obstime=self.utctime, location=self.observer._earthlocation))

    @property
    def data_dict(self) -> staraltParams:
        if not hasattr(self, 'data') or not self.data:
            raise ValueError("No data available for observability. Run set_target first.")
        return self.data.data_dict
    
    @property
    def is_observable(self) -> bool:
        if self.min_max_obstime is not None:
            return True

    @property
    def min_max_obstime(self):
        target_times = np.array(self.data.target_times)  # Time array
        visibility = np.array(self.data.visibility)  # Boolean visibility array

        # Get the current time
        current_time = self.utctime

        # Find indices where the target is observable
        visible_indices = np.where(visibility)[0]

        if len(visible_indices) == 0:
            return None  # No observable periods

        # Filter times that are in the future relative to the current time
        future_visible_indices = visible_indices[target_times[visible_indices] >= current_time]

        if len(future_visible_indices) == 0:
            return None  # No observable times remain

        # Get the next observable window
        start_idx = future_visible_indices[0]  # First available index
        end_idx = start_idx  # Expand until visibility stops

        while end_idx + 1 < len(visibility) and visibility[end_idx + 1]:
            end_idx += 1  # Extend the window

        return (target_times[start_idx], target_times[end_idx])

    def next_observable_night(self, search_days: int = 7):
        if not hasattr(self, 'data') or not self.data:
            print("No data available for observability. Run set_target first.")
            return None

        if self.is_observable:
            return self.min_max_obstime
        # Start checking for the next night
        utctime = self.utctime  # Current reference time
        added_days = 0
        while added_days < search_days:
            utctime += timedelta(days=1)  # Move to the next day
            self._set_night(utctime)  # Update observation night
            self.set_target(self.target_coord.ra.deg, self.target_coord.dec.deg)  # Recalculate observability
            
            if self.is_observable:
                return self.min_max_obstime  # Return next available time
            added_days +=1
        print(f"No observable time found within {search_days} days.")
        return None

    def _set_night(self, utctime: Optional[Union[dt, Time]] = None) -> obsNightParams:
        """
        Set the night for the given time.
        
        Parameters
        ----------
        utctime : datetime or Time, optional
            The time for which to set the night.
            
        Returns
        -------
        obsNightParams
            Object containing night-related time information.
        """
        if utctime is None:
            utctime = Time.now()
        if not isinstance(utctime, Time):
            utctime = Time(utctime)
        
        obsnight = obsNightParams()
        
        # Celestial information
        obsnight.sunrise_civil = self.observer.sun_risetime(utctime, horizon=0, mode='next')
        
        # Only continue if sunrise_civil is not None
        if obsnight.sunrise_civil is not None:
            obsnight.sunset_civil = self.observer.sun_settime(obsnight.sunrise_civil, mode='previous', horizon=0)
            obsnight.sunrise_nautical = self.observer.sun_risetime(obsnight.sunrise_civil, mode='previous', horizon=-6)
            obsnight.sunset_nautical = self.observer.sun_settime(obsnight.sunrise_civil, mode='previous', horizon=-6)
            obsnight.sunrise_astro = self.observer.sun_risetime(obsnight.sunrise_civil, mode='previous', horizon=-12)
            obsnight.sunset_astro = self.observer.sun_settime(obsnight.sunrise_civil, mode='previous', horizon=-12)
            obsnight.sunrise_night = self.observer.sun_risetime(obsnight.sunrise_civil, mode='previous', horizon=-18)
            obsnight.sunset_night = self.observer.sun_settime(obsnight.sunrise_civil, mode='previous', horizon=-18)
        
        self.tonight = obsnight

    def _get_skycoord(self, ra: Union[str, float], dec: Union[str, float]) -> SkyCoord:
        """
        Convert RA and Dec to SkyCoord object.
        
        Parameters
        ----------
        ra : str or float
            Right Ascension, if str in hms format (e.g., "10:20:30"),
            if float in decimal degrees.
        dec : str or float
            Declination, if str in dms format (e.g., "+20:30:40"),
            if float in decimal degrees.
        
        Returns
        -------
        SkyCoord
            SkyCoord object representing the coordinates.
            
        Raises
        ------
        ValueError
            If RA and Dec formats are not supported or coordinates are out of bounds.
        """
        try:
            # Check if RA and Dec are given as strings (sexagesimal format)
            if isinstance(ra, str) and isinstance(dec, str):
                # Interpret as sexagesimal format (e.g., "10:20:30", "+20:30:40")
                coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
            elif isinstance(ra, (float, int)) and isinstance(dec, (float, int)):
                # Convert to float for validation
                ra_deg = float(ra)
                dec_deg = float(dec)
                
                # Validate coordinate ranges BEFORE creating SkyCoord
                # RA validation: normalize to 0-360 range
                if ra_deg < 0:
                    ra_deg = ra_deg % 360  # Convert negative RA to positive equivalent
                elif ra_deg > 360:
                    ra_deg = ra_deg % 360  # Wrap RA > 360 to 0-360 range
                
                # DEC validation: must be within -90 to +90
                if not (-90.0 <= dec_deg <= 90.0):
                    raise ValueError(
                        f"Declination must be within -90° to +90°, got {dec_deg:.2f}°. "
                        f"Invalid coordinates: RA={ra_deg:.2f}°, DEC={dec_deg:.2f}°"
                    )
                
                # Create SkyCoord with validated coordinates
                coord = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)
                
            else:
                raise ValueError(
                    f"Unsupported RA and Dec format. "
                    f"Expected (float, float) or (str, str), got ({type(ra).__name__}, {type(dec).__name__})"
                )
            
            return coord
            
        except ValueError as e:
            # Re-raise ValueError with more context
            raise ValueError(f"Invalid coordinates: {str(e)}")
        except Exception as e:
            # Handle other potential errors (e.g., malformed sexagesimal strings)
            raise ValueError(f"Error creating sky coordinates: {str(e)}")

    def set_target(self, 
                     ra: Union[float, str], 
                     dec: Union[float, str], 
                     objname: Optional[str] = None, 
                     utctime: Optional[Union[dt, Time]] = None,
                     time_shift: Optional[int] = None,
                     target_minalt: float = 30, 
                     target_minmoonsep: float = 30) -> Dict[str, Any]:
        """
        Generate the data needed for the altitude plot.

        Parameters
        ----------
        ra : float or str
            Right Ascension of the target in degrees (float) or in hms format (str).
        dec : float or str
            Declination of the target in degrees (float) or in dms format (str).
        time_shift: int

        objname : str, optional
            Name of the target object.
        utctime : datetime or Time, optional
            The reference time. If None, current time is used.
        target_minalt : float, optional
            The minimum allowable altitude for observation (in degrees).
        target_minmoonsep : float, optional
            The minimum allowable moon separation (in degrees).

        Returns
        -------
        dict
            A dictionary containing all data required for plotting.
        """
        
        # Set the reference time
        if utctime is None:
            utctime = Time.now()
        elif not isinstance(utctime, Time):
            utctime = Time(utctime)
        
        self.logger.debug(f"Calculating visibility for time: {utctime.datetime}")
        
        # Clear any cached tonight data to ensure fresh calculation
        if hasattr(self, '_cached_tonight_date'):
            delattr(self, '_cached_tonight_date')
        
        # Calculate tonight for the specific date
        tonight = self.observer.tonight(time=utctime, horizon=-18*u.deg)
        
        # Validate that tonight calculation matches the input date
        expected_date = utctime.datetime.date()
        sunset_date = getattr(tonight, 'sunset_night', None)
        
        if sunset_date:
            sunset_date_only = sunset_date.datetime.date() if hasattr(sunset_date, 'datetime') else sunset_date.date()
            if sunset_date_only != expected_date:
                self.logger.warning(f"Night calculation mismatch: expected {expected_date}, got sunset on {sunset_date_only}")
                # Force recalculation with explicit date
                tonight = self.observer.tonight(time=utctime, horizon=-18*u.deg)
        
        self.logger.debug(f"Calculated tonight - Sunset: {getattr(tonight, 'sunset_night', None)}, "
                        f"Sunrise: {getattr(tonight, 'sunrise_night', None)}")

        # Get the sky coordinates of the target
        self._objname = objname
        self._target_coord = self._get_skycoord(ra, dec)

        # Define time range for plotting using Time arithmetic
        time_range_start = tonight.sunset_astro - 2*u.hour # type: ignore
        time_range_end = tonight.sunrise_astro + 2*u.hour # type: ignore

        # Optimize time axis generation - calculate time points more efficiently
        time_span_seconds = (time_range_end - time_range_start).sec
        # Use 5-minute intervals (300 seconds)
        num_points = int(time_span_seconds / 300) + 1
        time_axis = time_range_start + np.linspace(0, time_span_seconds, num_points)*u.second # type: ignore

        # Calculate altaz for moon, sun, and target - doing this in parallel would be ideal
        # but we'll optimize what we can within the current structure
        moon_altaz = self.observer.moon_altaz(time_axis)
        sun_altaz = self.observer.sun_altaz(time_axis)
        target_altaz = self._target_coord.transform_to(AltAz(obstime=time_axis, location=self.observer._earthlocation))

        # Now we can safely calculate separation
        target_moonsep = moon_altaz.separation(target_altaz)

        # Extract values using our helper functions
        target_alts = bumper.safe_to_list(bumper.safe_get_alt_value(target_altaz))
        moon_alts = bumper.safe_to_list(bumper.safe_get_alt_value(moon_altaz))
        sun_alts = bumper.safe_to_list(bumper.safe_get_alt_value(sun_altaz))
        target_moonsep_vals = bumper.safe_to_list(bumper.safe_get_separation_value(target_moonsep))
        
        # Optimize color determination using NumPy where possible
        try:
            # Extract time values for comparison - only do this once
            obs_times = bumper.safe_extract_times(target_altaz)
            
            # Pre-allocate the visibility array
            visibility = [False] * len(target_alts)
            
            # Only process if we have valid times
            if obs_times and len(obs_times) > 0:
                obs_times_array = np.array(obs_times)
                is_night_time = np.logical_and(obs_times_array >= tonight.sunset_night, 
                                            obs_times_array <= tonight.sunrise_night)
                is_above_min_alt = np.array(target_alts) > target_minalt
                is_min_moon_sep = np.array(target_moonsep_vals) > target_minmoonsep
                
                # Combine all conditions
                observable_mask = np.logical_and.reduce((is_night_time, is_above_min_alt, is_min_moon_sep))
                visibility = observable_mask.tolist()

        except (TypeError, AttributeError) as e:
            print(f"Warning: Error processing values for visibility: {e}")
            # Default to not observable
            visibility = [False] * len(target_alts)
        
        color_target = np.where(visibility, "g", "r").tolist()

        # Extract times in ISO format - do this once and cache the results
        moon_times = bumper.safe_extract_time_strings(moon_altaz)
        sun_times = bumper.safe_extract_time_strings(sun_altaz)
        target_times = bumper.safe_extract_time_strings(target_altaz)

        # Convert night times to datetime with proper None checks
        tonight_data: Dict[str, Optional[dt]] = {
            "sunset_night": bumper.safe_get_datetime(getattr(tonight, 'sunset_night', None)),
            "sunrise_night": bumper.safe_get_datetime(getattr(tonight, 'sunrise_night', None)),
            "sunset_civil": bumper.safe_get_datetime(getattr(tonight, 'sunset_civil', None)),
            "sunrise_civil": bumper.safe_get_datetime(getattr(tonight, 'sunrise_civil', None))
        }

        # Create the data dictionary - use the cached now value
        data_dict: Dict[str, Any] = {
            "objname": self.objname,
            "now_datetime": utctime.datetime,  
            "time_range_start": time_range_start.datetime,
            "time_range_end": time_range_end.datetime,
            "target_times": target_times,
            "target_alts": target_alts,
            "target_moonsep": target_moonsep_vals,
            "visibility": visibility,
            "color_target": color_target,
            "moon_times": moon_times,
            "moon_alts": moon_alts,
            "sun_times": sun_times,
            "sun_alts": sun_alts,
            "tonight": tonight_data,
            "target_minalt": target_minalt,
            "target_minmoonsep": target_minmoonsep   
        }
        
        self.data = staraltParams(data_dict=data_dict)
        return self.data



    def plot_staralt(self, data: Optional[Dict[str, Any]] = None, show_current_time: bool = True) -> None:
        """
        Plot the altitude data from the dictionary returned by set_target().

        Parameters
        ----------
        data : dict, optional
            The dictionary containing data required for plotting.
        show_current_time : bool, optional
            Whether to show the current time marker on the plot.
        """
        # Unpack data, using the instance's data_dict as a fallback
        if data is None:
            if not hasattr(self, 'data') or not self.data:
                raise ValueError("No data available for plotting. Run set_target first.")
            data = self.data

        # Extract data from dictionary with proper default handling
        objname = data.objname
        now_datetime = data.now_datetime
        moon_times = data.moon_times
        moon_alts = data.moon_alts
        target_times = data.target_times
        target_alts = data.target_alts
        color_target = data.color_target
        tonight = data.tonight
        target_minalt = data.target_minalt
        target_minmoonsep = data.target_minmoonsep
        
        # Extract and convert all datetime objects at once for better performance
        datetime_conversions = {
            "now": now_datetime,
            "sunset_night": tonight["sunset_night"],
            "sunrise_night": tonight["sunrise_night"],
            "sunset_civil": tonight["sunset_civil"],
            "sunrise_civil": tonight["sunrise_civil"]
        }
        
        # Check for required night times
        required_times = ["sunset_night", "sunrise_night", "sunset_civil", "sunrise_civil"]
        if not all(datetime_conversions[key] for key in required_times):
            raise ValueError("Required night times are missing from the data")
            
        # Bulk convert to numeric format
        numeric_times = {}
        for key, dt_value in datetime_conversions.items():
            numeric_times[key] = bumper.safe_date2num(dt_value)
            
        # Validate critical numeric values
        critical_keys = required_times
        if any(numeric_times[key] is None for key in critical_keys):
            raise ValueError("Failed to convert critical night times to numeric values")
        
        # Assign variables from the dictionary for cleaner code
        now_datetime_num = numeric_times["now"]
        sunset_night_num = numeric_times["sunset_night"]
        sunrise_night_num = numeric_times["sunrise_night"]  
        sunset_civil_num = numeric_times["sunset_civil"]
        sunrise_civil_num = numeric_times["sunrise_civil"]
        
        # Optimize time conversion by creating a conversion cache
        time_conversion_cache = {}
        
        # Convert moon and target times to numeric using the cached function
        moon_times_num = [bumper.safe_cached_time_to_num(t, time_conversion_cache) for t in moon_times]
        target_times_num = [bumper.safe_cached_time_to_num(t, time_conversion_cache) for t in target_times]
        
        # Calculate middle of the night once
        mid_night_num = sunset_night_num + 0.5 * (sunrise_night_num - sunset_night_num) if (sunset_night_num is not None and sunrise_night_num is not None) else 0.0
        
        # Determine title
        titlename = f'Altitude of {objname}' if objname else 'Altitude of the Target'
        
        # Find observable windows more efficiently
        if 'g' in color_target:
            # Use NumPy for faster array operations
            observable_indices = np.where(np.array(color_target) == 'g')[0]
            observable_start_idx = observable_indices[0] if len(observable_indices) > 0 else None
            observable_end_idx = observable_indices[-1] if len(observable_indices) > 0 else None
        else:
            observable_start_idx = None
            observable_end_idx = None
        
        # Initialize observation variables
        obs_start_time = None
        obs_end_time = None
        obs_start_num = None
        obs_end_num = None
        total_observable_hours = 0.0
        remaining_observable_hours = 0.0
        
        # Calculate observable time windows only if there are observable periods
        if observable_start_idx is not None and observable_end_idx is not None:
            if observable_start_idx < len(target_times) and observable_end_idx < len(target_times):
                # Get the start and end times
                start_time_str = target_times[observable_start_idx]
                end_time_str = target_times[observable_end_idx]
                
                # Convert with error handling
                try:
                    # Convert start time
                    if isinstance(start_time_str, str):
                        obs_start_time = datetime.fromisoformat(start_time_str)
                    else:
                        obs_start_time = start_time_str
                    
                    # Convert end time
                    if isinstance(end_time_str, str):
                        obs_end_time = datetime.fromisoformat(end_time_str)
                    else:
                        obs_end_time = end_time_str
                    
                    # Get numeric values for plotting
                    if obs_start_time and obs_end_time:
                        obs_start_num = bumper.safe_date2num(obs_start_time)
                        obs_end_num = bumper.safe_date2num(obs_end_time)
                        
                        # Calculate observation times
                        if obs_start_time and obs_end_time:
                            total_observable_hours = (obs_end_time - obs_start_time).total_seconds() / 3600.0
                            
                            if now_datetime and obs_end_time:
                                if now_datetime < obs_end_time:
                                    if now_datetime > obs_start_time:
                                        remaining_observable_hours = (obs_end_time - now_datetime).total_seconds() / 3600.0
                                    else:
                                        remaining_observable_hours = total_observable_hours
                except (ValueError, TypeError, IndexError) as e:
                    print(f"Warning: Error processing observable times: {e}")

        # Optimize the plotting section by preparing data structures first
        # Plotting - Create figure just once
        fig = plt.figure(dpi=300, figsize=(10, 4))
        plt.title(titlename, loc='center', y=1.05)
        
        # Add subtitle with observation criteria
        plt.figtext(0.515, 0.90, f"Criteria: Alt > {target_minalt}°, Moon sep > {target_minmoonsep}°", 
                    ha='center', va='center', fontsize=9, style='italic')

        # Ensure all arrays have matching lengths - do this once before plotting
        data_arrays = [
            (moon_times_num, moon_alts, "Moon times", "altitudes"),
            (target_times_num, target_alts, "Target times", "altitudes"),
            (target_times_num, color_target, "Target times", "colors")
        ]
        
        for time_array, data_array, time_name, data_name in data_arrays:
            if len(time_array) != len(data_array):
                print(f"Warning: {time_name} ({len(time_array)}) and {data_name} ({len(data_array)}) have different lengths")
                # Adjust to minimum length
                min_len = min(len(time_array), len(data_array))
                if time_array is moon_times_num and data_array is moon_alts:
                    moon_times_num = moon_times_num[:min_len]
                    moon_alts = moon_alts[:min_len]
                elif time_array is target_times_num and data_array is target_alts:
                    target_times_num = target_times_num[:min_len]
                    target_alts = target_alts[:min_len]
                elif time_array is target_times_num and data_array is color_target:
                    target_times_num = target_times_num[:min_len]
                    color_target = color_target[:min_len]

        # Batch together similar plotting operations
        
        # 1. Plot data points
        if moon_times_num is not None and moon_alts is not None:
            plt.scatter(moon_times_num, moon_alts, c='b', s=10, marker='.', label='Moon')

        if target_times_num is not None and target_alts is not None and color_target is not None:
            plt.scatter(target_times_num, target_alts, c=color_target, s=30, marker='*', label='Target')

        # 2. Fill regions and draw lines - use float conversion just once per value
        sunset_night_float = float(sunset_night_num)
        sunrise_night_float = float(sunrise_night_num)
        sunset_civil_float = float(sunset_civil_num)
        sunrise_civil_float = float(sunrise_civil_num)
        
        # Fill in regions
        plt.fill_betweenx([10, 90], sunset_night_float, sunrise_night_float, color='k', alpha=0.3)
        plt.fill_betweenx([10, 90], sunset_civil_float, sunrise_civil_float, color='k', alpha=0.1)
        
        # Draw vertical lines
        plt.axvline(x=sunrise_night_float, linestyle='-', c='k', linewidth=0.5)
        plt.axvline(x=sunset_night_float, linestyle='-', c='k', linewidth=0.5)
        
        # Fill region below minimum altitude
        plt.fill_between([sunset_night_float, sunrise_night_float], 0, float(target_minalt), color='r', alpha=0.3)

        # 3. Add text annotations - doing similar operations in batches
        text_annotations = [
            (sunset_night_float, 93, 'Night start'),
            (sunrise_night_float, 93, 'Night end'),
            (float(mid_night_num), 20, 'Observation limit')
        ]
        
        for x, y, text in text_annotations:
            color = 'darkred' if text == 'Observation limit' else 'k'
            plt.text(x, y, text, fontsize=10, ha='center', va='center', c=color)
        
        # 4. Add time markers with standard colors
        time_colors = {
            'current': 'purple',
            'start': 'darkgreen',
            'end': 'darkorange'
        }
        
        # Add current time marker
        if show_current_time and now_datetime_num is not None:
            now_float = float(now_datetime_num)
            plt.axvline(now_float, linestyle='--', c=time_colors['current'], linewidth=1.5, label='Current time')
            
            if now_datetime:
                plt.annotate(now_datetime.strftime("%H:%M"), 
                            xy=(now_float, 0), 
                            xycoords='data', 
                            xytext=(0, -10), 
                            textcoords='offset points',
                            color=time_colors['current'], 
                            fontsize=9,
                            ha='center', 
                            va='top')
        
        # Add observation window markers
        if (obs_start_num is not None and obs_end_num is not None):
            obs_start_float = float(obs_start_num)
            obs_end_float = float(obs_end_num)
            
            # Add vertical lines
            plt.axvline(obs_start_float, linestyle='-.', c=time_colors['start'], linewidth=1.5, label='Obs. start')
            plt.axvline(obs_end_float, linestyle='-.', c=time_colors['end'], linewidth=1.5, label='Obs. end')
            
            # Add time labels
            if obs_start_time:
                plt.annotate(obs_start_time.strftime("%H:%M"), 
                            xy=(obs_start_float, 0), 
                            xycoords='data', 
                            xytext=(0, -10), 
                            textcoords='offset points',
                            color=time_colors['start'], 
                            fontsize=9,
                            ha='center', 
                            va='top')
            
            if obs_end_time:
                plt.annotate(obs_end_time.strftime("%H:%M"), 
                            xy=(obs_end_float, 0), 
                            xycoords='data', 
                            xytext=(0, -10), 
                            textcoords='offset points',
                            color=time_colors['end'], 
                            fontsize=9,
                            ha='center', 
                            va='top')
        
        # 5. Create the information text
        time_info = []
        if now_datetime and show_current_time:
            time_info.append(f'Current Time: {now_datetime.strftime("%Y-%m-%d %H:%M:%S")} UTC')
        
        if obs_start_time is not None and obs_end_time is not None:
            time_info.append(f'Observable Period: {obs_start_time.strftime("%H:%M")} - {obs_end_time.strftime("%H:%M")} UTC')
            time_info.append('')  # Add empty line for spacing
            time_info.append(f'Total Observable Time: {total_observable_hours:.1f} hours')
            time_info.append(f'Remaining Observable Time: {remaining_observable_hours:.1f} hours')
        else:
            time_info.append('Target is not observable tonight')
        
        # Join once
        info_text = '\n'.join(time_info)
        
        # Calculate text position
        text_x = sunset_night_float - (sunrise_night_float - sunset_night_float) * 0.13
        
        # Add text box
        plt.text(text_x, 85, info_text, 
                fontsize=9, ha='left', va='top', c='k', 
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'),
                linespacing=1.3)

        # 6. Final plot setup - do this once at the end
        plt.xlim(sunset_civil_float, sunrise_civil_float)
        plt.ylim(10, 90)
        plt.legend(loc='upper right')
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(DateFormatter('%m-%d %H'))
        plt.xlabel('UTC Time [mm-dd hh]')
        plt.ylabel('Altitude [degrees]')
        plt.grid()
        plt.xticks(rotation=45)
        
        # Add bottom margin
        plt.subplots_adjust(bottom=0.18)