from astroplan import FixedTarget, is_event_observable, is_observable, is_always_observable
from astroplan import AltitudeConstraint, MoonSeparationConstraint, AtNightConstraint
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
import numpy as np
import seaborn as sns
import datetime
from typing import List
import json
from pathlib import Path
# TCSpy modules
import warnings
from astropy.utils.exceptions import AstropyWarning


from .configuration import Configuration
from .observatory import mainObserver

warnings.filterwarnings("ignore", category=AstropyWarning)

class Targets:
    """
    A class representing multiple astronomical targets for observation.

    Parameters
    ----------s
    observer : mainObserver
        An instance of mainObserver representing the observer.
    ra : numpy.array
        An array containing the right ascension values of the targets, in degrees.
    dec : numpy.array
        An array containing the declination values of the targets, in degrees.
    name : numpy.array, optional
        An array containing the names of the targets. Default is None.
        
    Attributes
    ----------
    ra : numpy.array
        An array of right ascension coordinates of the targets in degrees.
    dec : numpy.array
        An array of declination coordinates of the targets in degrees.
    coordinate : SkyCoord
        The astropy SkyCoord object representing the coordinates of the targets.
    target_astroplan : list
        A list of astroplan FixedTarget objects representing the targets.
    name : numpy.array
        An array of names corresponding to the targets.

    Methods
    -------
    rts_date(year=None, time_grid_resolution=3)
        Calculate the rise, transit, and set times of the targets for each day of the specified year.
    is_observable
    
    is_ever_observable(utctime=None, time_grid_resolution=1*u.hour)
        Determines whether the targets are observable during the specified time.
    is_always_observable(utctimes=None)
        Determines whether the targets are always observable during the specified time.
    is_event_observable(utctimes=None)
        Determines whether the targets are observable at the specified time or at the current time.
    altaz(utctime=None)
        Calculate the alt-az coordinates of the targets for the given time(s) in UTC.
    risetime(utctime=None, mode='nearest', horizon=30, n_grid_points=50)
        Calculate the next rise time of the targets as seen by the observer.
    settime(utctime=None, mode='nearest', horizon=30, n_grid_points=50)
        Calculate the time when the targets set below the horizon.
    meridiantime(utctime=None, mode='nearest', n_grid_points=50)
        Calculate the time at which the targets pass through the observer's meridian.
    hourangle(utctime=None)
        Calculate the hour angle of the targets for the given time(s) in UTC.
    """
    
    def __init__(self,
                 name_list : np.array,
                 ra_list : np.array = [],
                 dec_list : np.array = [],
                 observer : mainObserver = None,
                 ):
        config_path_list = [f'config/target.config',
                            f'config/nightsession.config']
        self.config = Configuration(config_filenames=config_path_list)
        
        if observer is None:
            self.observer = mainObserver()
        else:
            self.observer = observer
        self._observer = self.observer._observer
        
        if len(ra_list) == 0 and len(dec_list) == 0:
            name_list = list(sorted(name_list))
            ra_list = []; dec_list = []
            for name in name_list:
                coord = SkyCoord.from_name(name)
                ra_list.append(coord.ra.degree)
                dec_list.append(coord.dec.degree)
        
        self.ra = np.atleast_1d(ra_list)
        self.dec = np.atleast_1d(dec_list)     
        self.name = name_list
        
        self._coordinate = self._get_coordinate_radec(ra = ra_list, dec = dec_list)
        self._target = self._get_target(self._coordinate, name_list)
        self._constraints = self._get_constraints() # Contrasints from the configuration file
        self.n_targets = len(self._coordinate) if len(self.ra) > 1 else 1
        
    def __repr__(self):
        txt = f'Targets[n_targets = {self.n_targets}]'
        return txt
    
    def rts_date(self,
                 year : int = None,
                 time_grid_resolution : float = 3 # timegrid for checking the observability 
                 ):
        """
        Calculate the rise, transit, and set times of the targets for each day of the specified year.

        Parameters
        ----------
        year : int, optional
            The year for which to calculate the rise, transit, and set times. Default is None.
        time_grid_resolution : float, optional
            The time grid resolution for checking the observability. Default is 3.

        Returns
        -------
        numpy.array
            An array containing the calculated rise, transit, and set times of the targets for each day of the year.
        """
        # If start_date & end_date are not specified, defaults to current year
        if year == None:
            year = Time.now().datetime.year
        start_date = Time(datetime.datetime(year = year, month = 1, day = 1))
        end_date = Time(datetime.datetime(year = year+1 , month = 1, day = 1))

        expanded_arrays_observability = []
        expanded_arrays_altitude_midnight = []
        expanded_arrays_date = []
        current_date = start_date
        while current_date <= end_date:
            print(f"Calculating observability of the {len(self._coordinate)} targets on {current_date.strftime('%Y-%m-%d')}")
            midnight = Time((self._observer.tonight(current_date)[0].jd + self._observer.tonight(current_date)[1].jd )/2, format = 'jd')
            alt_at_midnight = self.altaz(midnight).alt.value
            expanded_arrays_altitude_midnight.append(alt_at_midnight)
            observablity = self.is_ever_observable(current_date, None, time_grid_resolution= time_grid_resolution * u.hour)
            expanded_arrays_observability.append(observablity)
            expanded_arrays_date.append(current_date.datetime)
            current_date += 1 * u.day

        observablity_array = np.array(expanded_arrays_observability).T
        altitude_array = np.array(expanded_arrays_altitude_midnight).T
        date_array = np.array(expanded_arrays_date)
        
        all_observability = []
        # Find the indices where the value changes
        for target_observability, target_altitude in zip(observablity_array, altitude_array):
            if all(target_observability):
                risedate = 'Always'
                setdate = 'Always'
                bestdate = date_array[np.argmax(target_altitude)]
            elif all(~target_observability):
                risedate = 'Never'
                setdate = 'Never'
                bestdate = 'Never'
            else:
                # Find index where observability False to True
                risedate_index = np.where(np.diff(target_observability.astype(int)) == 1)[0] + 1
                risedate = date_array[risedate_index[0]]
                # Find index where observability True to False
                setdate_index = np.where(np.diff(target_observability.astype(int)) == -1)[0] + 1
                setdate = date_array[setdate_index[0]]
                bestdate = date_array[np.argmax(target_altitude)]
            all_observability.append((risedate, bestdate, setdate))
        
        return np.array(all_observability), observablity_array, altitude_array, date_array
        
    def moon_sep(self,
                 utctime : datetime or Time or np.array = None):
        """
        Calculate the separation between the Moon and the target.

        Parameters
        ----------
        utctime : datetime or Time, optional
            The time at which to calculate the separation. If not provided, the current time will be used.

        Returns
        -------
        moonsep : astropy.coordinates.Angle
            The separation between the Moon and the target.
        """
            
        if utctime is None:
            utctime = Time.now()
        if not isinstance(utctime, Time):
            utctime = Time(utctime)
        moon_altaz = self.observer.moon_altaz(utctime)
        target_altaz = self.altaz(utctime)
        moonsep = np.round(moon_altaz.separation(target_altaz),2)
        return moonsep.value
    
    def is_observable(self,
                      utctime : datetime or Time or np.array = None) -> bool:
        """
        Determines whether the targets are observable at the specified time.
        """
        if utctime is None:
            utctime = Time.now()
        if not isinstance(utctime, Time):
            utctime = Time(utctime)
        return is_event_observable(constraints = self._constraints, observer = self._observer, target = self._target, times = utctime)
   
    def is_ever_observable(self,
                           utctime_start : datetime or Time = None,
                           utctime_end : datetime or Time = None,
                           time_grid_resolution = 1 * u.hour) -> List[bool]:
        """
        Determines whether the targets are observable during the specified time.

        Parameters
        ----------
        utctime : datetime or Time, optional
            The time at which to check observability. Defaults to the current time.
        time_grid_resolution : astropy.units.Quantity, optional
            The time grid resolution for checking the observability. Default is 1 hour.

        Returns
        -------
        List[bool]
            A list of boolean values indicating whether each target is observable during the specified time.

        Raises
        ------
        TypeError
            If the provided time is not a valid datetime or Time object.
        """
        # Ensure both start and end times are Time objects
        if utctime_start is not None and not isinstance(utctime_start, Time):
            utctime_start = Time(utctime_start)
        if utctime_end is not None and not isinstance(utctime_end, Time):
            utctime_end = Time(utctime_end)
  
        # Handle case where both times are None
        if utctime_start is None and utctime_end is None:
            tonight = self.observer.tonight(Time.now())
            starttime, endtime = tonight[0], tonight[1]
        # Handle case where one time is None
        elif utctime_start is None:
            tonight = self.observer.tonight(utctime_end)
            starttime, endtime = tonight[0], utctime_end
        elif utctime_end is None:
            tonight = self.observer.tonight(utctime_start)
            starttime, endtime = utctime_start, tonight[1]
        # Handle case where both times are provided
        else:
            starttime, endtime = utctime_start, utctime_end              

        time_range = [starttime, endtime]
        return is_observable(constraints = self._constraints, observer = self._observer, targets = self._target, time_range = time_range, time_grid_resolution = time_grid_resolution)
    
    def is_always_observable(self,
                             utctimes : datetime or Time or np.array = None) -> bool:
        """
        Determines whether the targets are always observable during the specified time.

        Parameters
        ----------
        utctimes : datetime or Time or numpy.array, optional
            The time at which to check observability. Defaults to the current time.

        Returns
        -------
        bool
            True if all targets are always observable, False otherwise.

        Raises
        ------
        TypeError
            If the provided time is not a valid datetime, Time, or numpy.array object.
        """
        if utctimes is None:
            utctimes = Time.now()
        if not isinstance(utctimes, Time):
            utctimes = Time(utctimes)
        return is_always_observable(constraints = self._constraints, observer = self._observer, targets = self._target, times = utctimes)
    
    def altaz(self,
              utctime : datetime or Time or np.array = None) -> SkyCoord:
        """
        Calculate the alt-az coordinates of the targets for the given time(s) in UTC.

        Parameters
        ----------
        utctime : datetime or Time or numpy.array, optional
            The time(s) to calculate the alt-az coordinates for, in UTC. If not provided, the current time will be used.

        Returns
        -------
        SkyCoord
            The alt-az coordinates of the targets at the specified time(s).

        Raises
        ------
        TypeError
            If the provided time is not a valid datetime, Time, or numpy.array object.
        """
        if utctime is None:
            utctime = Time.now()
        if not isinstance(utctime, Time):
            utctime = Time(utctime)
        result = self._observer.altaz(utctime, target=self._target, grid_times_targets=True)
        if self.n_targets == 1:
            return result[0]
        return result
    
    def risetime(self,
                 utctime : datetime or Time = None ,
                 mode : str = 'nearest',
                 horizon : float = 30,
                 n_grid_points : int = 50) -> Time:
        """
        Calculate the next rise time of the targets as seen by the observer.

        Parameters
        ----------
        utctime : datetime or Time, optional
            The time to start searching for the next rise time. If not provided, the current time will be used.
        mode : str, optional
            The method used to determine the rise time. Possible values are 'next' (the next rise time), 'previous' (the previous rise time), or 'nearest' (the nearest rise time). Default is 'next'.
        horizon : float, optional
            The altitude of the horizon, in degrees. Default is 30.
        n_grid_points : int, optional
            The number of grid points to use in the interpolation. Default is 50.

        Returns
        -------
        Time
            The rise time of the targets as seen by the observer.

        """
        if utctime == None:
            utctime = Time.now()
        if not isinstance(utctime, Time):
            utctime = Time(utctime)
        return self._observer.target_rise_time(utctime, target = self._target, which = mode, horizon = horizon*u.deg, n_grid_points = n_grid_points)
    
    def settime(self,
                utctime : datetime or Time or np.array = None,
                mode : str = 'nearest',
                horizon : float = 30,
                n_grid_points : int = 50) -> Time:
        """
        Calculate the time when the targets set below the horizon.

        Parameters
        ----------
        utctime : datetime or Time or numpy.array, optional
            The time to use as the reference time for the calculation, by default the current time.
        mode : str, optional
            Set to 'nearest', 'next' or 'previous', by default 'nearest'.
        horizon : float, optional
            The altitude of the horizon in degrees. Default is 30.
        n_grid_points : int, optional
            The number of grid points to use in the interpolation. Default is 50.

        Returns
        -------
        Time
            The time when the targets set below the horizon.

        """
        if utctime is None:
            utctime = Time.now()
        if not isinstance(utctime, Time):
            utctime = Time(utctime)
        return self._observer.target_set_time(utctime, self._target, which = mode, horizon = horizon*u.deg , n_grid_points = n_grid_points)
    
    def meridiantime(self,
                     utctime : datetime or Time or np.array = None,
                     mode : str = 'nearest',
                     n_grid_points : int = 50) -> Time:
        """
        Calculate the time at which the targets pass through the observer's meridian.

        Parameters
        ----------
        utctime : datetime or Time or numpy.array, optional
            The time at which to calculate the meridian transit time. If not provided, the current time will be used.
        mode : str, optional
            Set to 'nearest', 'next' or 'previous', by default 'nearest'.
        n_grid_points : int, optional
            The number of grid points to use in the interpolation. Default is 50.

        Returns
        -------
        Time
            The time at which the targets pass through the observer's meridian.

        """
        if utctime is None:
            utctime = Time.now()
        if not isinstance(utctime, Time):
            utctime = Time(utctime)
        return self._observer.target_meridian_transit_time(utctime, self._target, which = mode, n_grid_points = n_grid_points)
    
    def hourangle(self,
                  utctime : datetime or Time or np.array = None):
        """
        Calculate the hour angle of the targets for the given time(s) in UTC.

        Parameters
        ----------
        utctime : datetime or Time or numpy.array, optional
            The time(s) to calculate the hour angle of the targets for, in UTC. If not provided, the current time will be used.

        Returns
        -------
        astropy.coordinates.Angle
            The hour angle of the targets at the specified time(s).

        Raises
        ------
        ValueError
            If no target is specified for hourangle.

        """
        if utctime is None:
            utctime = Time.now()
        if not isinstance(utctime, Time):
            utctime = Time(utctime)
        if not isinstance(self._target, FixedTarget):
            raise ValueError('No target is specified for hourangle')
        return self._observer.target_hour_angle(utctime, self._target)
    
    def plot_visibility(self,
                        utctime : datetime or Time or np.array = None,
                        show : bool = True,
                        save_path: str = None):
        """
        Creates a plot of the altitude and azimuth of a celestial object.
        
        Parameters
        ----------
        utctime : datetime or Time or np.array, optional
            The time(s) for which to calculate the altitude and azimuth of the celestial object. 
            If not provided, the current time is used.
        save_path: str, optional
            The path to save the plot. If not provided, the plot will not be saved.
        """
        import matplotlib.pyplot as plt
        now = Time.now()
        if utctime is None:
            utctime = Time.now()
        if not isinstance(utctime, Time):
            utctime = Time(utctime)

        astro_sunsettime  = self.observer.sun_settime(utctime, horizon=self.config.sunalt_observation)
        astro_sunrisetime = self.observer.sun_risetime(astro_sunsettime, horizon=self.config.sunalt_observation, mode='next')
        sunsettime = self.observer.sun_settime(utctime, horizon=0)
        sunrisetime = self.observer.sun_risetime(sunsettime, horizon=0, mode='next')

        time_range_start = sunsettime.datetime - datetime.timedelta(hours=2)
        time_range_end   = sunrisetime.datetime + datetime.timedelta(hours=2)

        time_axis = np.arange(time_range_start, time_range_end, datetime.timedelta(minutes=5))

        moon_sep = self.moon_sep(time_axis)
        moon_altaz = self.observer.moon_altaz(time_axis)
        sun_altaz = self.observer.sun_altaz(time_axis)
        target_altaz = self.altaz(time_axis)

        plt.figure(dpi=300, figsize=(10, 4))

        if (now.datetime < time_range_end + datetime.timedelta(hours=3)) & (now.datetime > time_range_start - datetime.timedelta(hours=3)):
            plt.axvline(now.datetime, linestyle='--', c='r', label='Now')

        plt.plot(moon_altaz.obstime.datetime, moon_altaz.alt.value, c='#8a7f8d', lw=0.7, label='Moon')
        plt.plot(sun_altaz.obstime.datetime, sun_altaz.alt.value, c='#ffcd3a', lw=0.7, label='Sun')

        if self.n_targets == 1:
            targets_iter = [(target_altaz, moon_sep, self._target, 'Target', 'k')]
        else:
            targets_iter = []
            colors = sns.cubehelix_palette(n_colors=self.n_targets, start=0, rot=-0.3, gamma=1.5, hue=1.0, light=0.80, dark=0.20, reverse=False, as_cmap=False)
            for i in range(self.n_targets):
                coord = self._coordinate[i]
                name = self.name[i] if self.name is not None else f"Target {i}"
                t_obj = self._get_target(coord, name)
                targets_iter.append((target_altaz[i], moon_sep[i], t_obj, name, colors[i]))

        for t_altaz, t_sep, t_obj, t_name, t_color in targets_iter:
            observable = is_event_observable(constraints=self._constraints, observer=self._observer, target=t_obj, times=Time(time_axis))
            if np.ndim(observable) > 1: observable = observable.flatten()
            observable = np.array(observable, dtype=bool)
            
            plt.plot(t_altaz.obstime.datetime, t_altaz.alt.value, c=t_color, alpha=0.3, label=None, lw=1.2)
            
            if np.any(observable):
                plt.plot(t_altaz.obstime.datetime[observable], t_altaz.alt.value[observable], c=t_color, alpha=1.0, label=t_name, lw=1.2)
            
            
            step = 12
            for i in range(0, len(time_axis), step):
                x = t_altaz.obstime.datetime[i]
                y = t_altaz.alt.value[i]
                sep = t_sep[i]
                if y > 10:
                    plt.text(x, y + 2, f'{sep:.0f}°', fontsize=7, ha='center', va='bottom', color='navy')

        plt.fill_betweenx([10, 90], astro_sunsettime.datetime, astro_sunrisetime.datetime, color="#022660", alpha=0.1)
        plt.fill_betweenx([10, 90], sunsettime.datetime, sunrisetime.datetime, color="#6D54A9", alpha=0.1)

        plt.axvline(x=astro_sunrisetime.datetime, linestyle='-', c='k', linewidth=0.5)
        plt.axvline(x=astro_sunsettime.datetime, linestyle='-', c='k', linewidth=0.5)
        plt.axvline(x=sunrisetime.datetime, linestyle='--', c='k', linewidth=0.5)
        plt.axvline(x=sunsettime.datetime, linestyle='--', c='k', linewidth=0.5)

        plt.xlim(time_range_start - datetime.timedelta(hours=1), time_range_end + datetime.timedelta(hours=1))
        plt.ylim(10, 90)
        plt.legend(loc=1)
        plt.xlabel('UTC [mm-dd hh]')
        plt.ylabel('Altitude [deg]')
        plt.grid()
        plt.xticks(rotation=45)
        # --- after all plotting commands ---

        # Build constraint text from config
        constraint_text = ['CONSTRAINTS', '-------------------------']
        if self.config.altitude_lower is not None:
            constraint_text.append(f"Min Alt: {self.config.altitude_lower}°")
        if self.config.altitude_upper is not None:
            constraint_text.append(f"Max Alt: {self.config.altitude_upper}°")
        if self.config.moonsep_lower is not None:
            constraint_text.append(f"Min Moon Sep: {self.config.moonsep_lower}°")

        # Join as multi-line string
        constraint_str = "\n".join(constraint_text)

        # Place text box inside the axes
        plt.gca().text(
            0.02, 0.90,             # position (axes fraction: right side, vertically centered)
            constraint_str,
            transform=plt.gca().transAxes,
            fontsize=7,
            va="center",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="gray")
        )
        if save_path is not None:
            save_path = Path(save_path)
            if not save_path.parent.exists():
                save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        
        
    def plot_optimal(self,
                     utctime : datetime or Time or np.array = None,
                     show : bool = True,
                     save_path: str = None):
        """
        Creates a plot of the altitude and azimuth of a celestial object.
        
        Parameters
        ----------
        utctime : datetime or Time or np.array, optional
            The time(s) for which to calculate the altitude and azimuth of the celestial object. 
            If not provided, the current time is used.
        save_path: str, optional
            The path to save the plot. If not provided, the plot will not be saved.
        """
        import matplotlib.pyplot as plt
        now = Time.now()
        if utctime is None:
            utctime = Time.now()
        if not isinstance(utctime, Time):
            utctime = Time(utctime)

        rts_date, observabilities, altitudes, dates = self.rts_date()

        plt.figure(dpi=300, figsize=(10, 4))

        colors = sns.cubehelix_palette(n_colors=self.n_targets, start=0, rot=-0.3, gamma=1.5, hue=1.0, light=0.80, dark=0.20, reverse=False, as_cmap=False)
        from matplotlib import colormaps
        import numpy as np
        cmap = colormaps.get_cmap("viridis")
        colors = cmap(np.linspace(0.0, 1.0, self.n_targets))

        for i, (observable, alt, color) in enumerate(zip(observabilities, altitudes, colors)):
            observable = np.asarray(observable)
            alt = np.asarray(alt)
            plt.plot(dates, alt, color=color, alpha=0.3)
            plt.plot(dates[observable], alt[observable], color=color, alpha=1, label=self.name[i])
        
        
        plt.ylim(10, 90)
        plt.legend(loc=1)
        plt.xlabel('UTC [mm-dd hh]')
        plt.ylabel('Altitude [deg]')
        plt.grid()
        plt.xticks(rotation=45)
        
        
        if save_path is not None:
            save_path = Path(save_path)
            if not save_path.parent.exists():
                save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def _get_coordinate_radec(self,
                              ra,
                              dec,
                              frame : str = 'icrs') -> SkyCoord:
        return SkyCoord(ra = ra, dec = dec, frame = frame, unit = (u.deg, u.deg))

    def _get_target(self,
                    coord,
                    target_name : str = '') -> FixedTarget:
        return FixedTarget(coord = coord, name = target_name)

    def _get_constraints(self) -> list:
        constraint_all = []
        if (self.config.altitude_lower != None) & (self.config.altitude_upper != None):
            constraint_altitude = AltitudeConstraint(min = self.config.altitude_lower * u.deg, max = self.config.altitude_upper * u.deg, boolean_constraint = True)
            constraint_all.append(constraint_altitude)
        if self.config.moonsep_lower != None:
            constraint_moonsep = MoonSeparationConstraint(min = self.config.moonsep_lower * u.deg, max = None)
            constraint_all.append(constraint_moonsep)
        if self.config.sunalt_observation != None:
            constraint_sunalt_observation = AtNightConstraint(max_solar_altitude = self.config.sunalt_observation * u.deg)
            constraint_all.append(constraint_sunalt_observation)
        
        return constraint_all