"""
Core observer class for astronomical observations
Optimized and streamlined from original mainobserver.py
"""

import numpy as np
from datetime import datetime
from typing import Union, Optional, Tuple

from astropy.coordinates import EarthLocation, get_sun, get_body, SkyCoord, AltAz
from astropy.time import Time
from astropy import units as u
from astroplan import Observer as AstroplanObserver
import pytz


class Observer:
    """
    Streamlined observer class for astronomical observations from Chile.
    """
    
    def __init__(self,
                 longitude: float = -70.7804,
                 latitude: float = -30.4704,
                 elevation: float = 1580,
                 timezone: str = "America/Santiago",
                 name: str = "7DT"):
        """
        Initialize observer with Chilean observatory defaults.
        
        Parameters
        ----------
        longitude : float
            Observatory longitude in degrees
        latitude : float
            Observatory latitude in degrees
        elevation : float
            Observatory elevation in meters
        timezone : str
            Observatory timezone string
        name : str
            Observer name
        observatory : str
            Observatory name
        """
        self.latitude = latitude * u.deg
        self.longitude = longitude * u.deg
        self.elevation = elevation * u.m
        self.name = name
        self.timezone = pytz.timezone(timezone)
        
        # Create earth location and astroplan observer
        self.location = EarthLocation.from_geodetic(
            lat=self.latitude, 
            lon=self.longitude, 
            height=self.elevation
        )
        self.astroplan_observer = AstroplanObserver(
            location=self.location,
            name=name,
            timezone=self.timezone
        )
    
    def now(self) -> Time:
        """Get current UTC time."""
        return Time.now()
    
    def tonight(self, time: Optional[Time] = None, horizon: float = -18) -> Tuple[Time, Time]:
        """
        Get tonight's astronomical night window.
        
        Parameters
        ----------
        time : Time, optional
            Reference time (default: now)
        horizon : float
            Sun altitude for astronomical night (default: -18 degrees)
        
        Returns
        -------
        tuple
            (sunset_time, sunrise_time) for astronomical night
        """
        if time is None:
            time = self.now()
        
        return self.astroplan_observer.tonight(time, horizon=horizon*u.deg)
    
    def get_twilight_times(self, time: Optional[Time] = None) -> dict:
        """
        Get all twilight times for tonight.
        
        Parameters
        ----------
        time : Time, optional
            Reference time (default: now)
        
        Returns
        -------
        dict
            Dictionary with all twilight times:
            - sunset_civil, sunrise_civil (sun at 0°)
            - sunset_nautical, sunrise_nautical (sun at -6°)
            - sunset_astro, sunrise_astro (sun at -12°)
            - sunset_night, sunrise_night (sun at -18°)
        """
        if time is None:
            time = self.now()
        
        # Get astronomical night as reference
        sunset_night, sunrise_night = self.tonight(time, horizon=-18)
        
        # Calculate all twilight times
        twilight_times = {
            'sunset_night': sunset_night,
            'sunrise_night': sunrise_night,
            'sunset_astro': self.astroplan_observer.sun_set_time(
                sunset_night, which='nearest', horizon=-12*u.deg
            ),
            'sunrise_astro': self.astroplan_observer.sun_rise_time(
                sunset_night, which='next', horizon=-12*u.deg
            ),
            'sunset_nautical': self.astroplan_observer.sun_set_time(
                sunset_night, which='nearest', horizon=-6*u.deg
            ),
            'sunrise_nautical': self.astroplan_observer.sun_rise_time(
                sunset_night, which='next', horizon=-6*u.deg
            ),
            'sunset_civil': self.astroplan_observer.sun_set_time(
                sunset_night, which='nearest', horizon=0*u.deg
            ),
            'sunrise_civil': self.astroplan_observer.sun_rise_time(
                sunset_night, which='next', horizon=0*u.deg
            )
        }
        
        return twilight_times
    
    def sun_altaz(self, times: Union[Time, np.ndarray]) -> AltAz:
        """
        Calculate sun altitude and azimuth.
        
        Parameters
        ----------
        times : Time or array of Times
            Times for calculation
        
        Returns
        -------
        AltAz
            Sun's altitude and azimuth
        """
        if not isinstance(times, Time):
            times = Time(times)
        return self.astroplan_observer.sun_altaz(times)
    
    def moon_altaz(self, times: Union[Time, np.ndarray]) -> AltAz:
        """
        Calculate moon altitude and azimuth.
        
        Parameters
        ----------
        times : Time or array of Times
            Times for calculation
        
        Returns
        -------
        AltAz
            Moon's altitude and azimuth
        """
        if not isinstance(times, Time):
            times = Time(times)
        return self.astroplan_observer.moon_altaz(times)
    
    def moon_illumination(self, time: Optional[Time] = None) -> float:
        """
        Get moon illumination fraction.
        
        Parameters
        ----------
        time : Time, optional
            Time for calculation (default: now)
        
        Returns
        -------
        float
            Moon illumination fraction (0-1)
        """
        if time is None:
            time = self.now()
        return self.astroplan_observer.moon_illumination(time)
    
    def target_altaz(self, ra: float, dec: float, times: Union[Time, np.ndarray]) -> AltAz:
        """
        Calculate target altitude and azimuth.
        
        Parameters
        ----------
        ra : float
            Right ascension in degrees
        dec : float
            Declination in degrees
        times : Time or array of Times
            Times for calculation
        
        Returns
        -------
        AltAz
            Target's altitude and azimuth
        """
        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        if not isinstance(times, Time):
            times = Time(times)
        
        return coord.transform_to(AltAz(obstime=times, location=self.location))
    
    def is_night(self, time: Optional[Time] = None, horizon: float = -18) -> bool:
        """
        Check if it's astronomical night.
        
        Parameters
        ----------
        time : Time, optional
            Time to check (default: now)
        horizon : float
            Sun altitude threshold (default: -18 degrees)
        
        Returns
        -------
        bool
            True if astronomical night
        """
        if time is None:
            time = self.now()
        
        return self.astroplan_observer.is_night(time, horizon=horizon*u.deg)
    
    def get_night_grid(self, time: Optional[Time] = None, 
                       horizon: float = -18,
                       time_resolution: int = 5) -> np.ndarray:
        """
        Generate time grid for the night.
        
        Parameters
        ----------
        time : Time, optional
            Reference time (default: now)
        horizon : float
            Sun altitude for night definition
        time_resolution : int
            Time resolution in minutes
        
        Returns
        -------
        np.ndarray
            Array of Time objects covering the night
        """
        if time is None:
            time = self.now()
        
        # Get tonight's window
        sunset, sunrise = self.tonight(time, horizon)
        
        # Extend slightly for plotting
        start_time = sunset - 2*u.hour
        end_time = sunrise + 2*u.hour
        
        # Generate time grid
        duration_minutes = (end_time - start_time).to(u.minute).value
        n_points = int(duration_minutes / time_resolution) + 1
        
        return start_time + np.linspace(0, duration_minutes, n_points) * u.minute
