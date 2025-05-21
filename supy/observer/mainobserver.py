
#%%
from astropy.coordinates import EarthLocation, get_sun
from datetime import datetime
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astroplan import Observer
from typing import Union, Optional, List
from astropy import units as u
import pytz
import inspect
import numpy as np
#%%
class mainObserver:
    """
    Class for observing astronomical objects and events from a specific location on Earth.

    Parameters
    ==========
    1. OBSERVER_LATITUDE : str
        The latitude of the observer's location in degrees.
    2. OBSERVER_LONGITUDE : str
        The longitude of the observer's location in degrees.
    3. OBSERVER_ELEVATION : str
        The elevation of the observer's location in meters.
    4. OBSERVER_TIMEZONE : str
        The timezone of the observer's location, in the format 'Area/Location'.
    5. OBSERVER_NAME : str, optional
        The name of the observer.
    6. OBSERVER_OBSERVATORY : str, optional
        The name of the observatory.
    **kwargs : optional
        Additional keyword arguments to pass to the Observer object.

    Methods
    =======
    1. get_obsinfo() -> dict
        Returns a dictionary containing the observer's information.
    2. localtime(utctime: datetime = None) -> datetime
        Converts the provided UTC time to the observer's local time.
    4. now() -> astropy.time.Time
        Returns the current UTC time.
    5. is_night(time: datetime or Time = None) -> bool
        Returns True if it is night at the observer's location at the provided UTC time.
    6. tonight(time: datetime or Time = None, horizon: float = -18) -> tuple
        Calculates the start and end times of tonight at the observer's location, starting from the provided UTC time.
    7. sun_radec(time: datetime or Time = None) -> astropy.coordinates.SkyCoord
        Calculates the RA and Dec of the Sun at the provided UTC time.
    8. sun_altaz(time: datetime or Time = None) -> astropy.coordinates.AltAz
        Calculates the altitude and azimuth of the Sun at the observer's location at the provided UTC time.
    9. sun_risetime(time: datetime or Time = None, mode: str = 'nearest', horizon: float = -18) -> astropy.time.Time
        Calculates the next rise time of the Sun at the observer's location, starting from the provided UTC time.
    10. sun_settime(time: datetime or Time = None, mode: str = 'nearest', horizon: float = -18) -> astropy.time.Time
        Calculates the next set time of the Sun at the observer's location, starting from the provided UTC time.
    11. moon_radec(time: datetime or Time = None) -> astropy.coordinates.SkyCoord
        Calculates the RA and Dec of the Moon at the provided UTC time.
    12. moon_altaz(time: datetime or Time = None) -> astropy.coordinates.AltAz
        Calculates the altitude and azimuth of the Moon at the observer's location at the provided UTC time.
    13. moon_risetime(time: datetime or Time = None, mode: str = 'nearest', horizon: float = -18) -> astropy.time.Time
        Calculates the next rise time of the Moon at the observer's location, starting from the provided UTC time.
    14. moon_settime(time: datetime or Time = None, mode: str = 'nearest', horizon: float = -18) -> astropy.time.Time
        Calculates the next set time of the Moon at the observer's location, starting from the
    """
    
    def __init__(self,
                OBSERVER_LONGITUDE: Union[str, float, int] = -70.7804,
                OBSERVER_LATITUDE: Union[str, float, int] = -30.4704,
                OBSERVER_ELEVATION: Union[str, float, int] = 1580,
                OBSERVER_TIMEZONE: str = "America/Santiago",
                OBSERVER_NAME: str = "Hyeonho Choi",
                OBSERVER_OBSERVATORY: str = "7DT"
                ):
        
        # Convert to float in case strings were provided
        self._latitude = float(OBSERVER_LATITUDE) * u.deg
        self._longitude = float(OBSERVER_LONGITUDE) * u.deg
        self._elevation = float(OBSERVER_ELEVATION) * u.m
        self._name = OBSERVER_NAME
        self._observatory = OBSERVER_OBSERVATORY
        self._timezone = pytz.timezone(OBSERVER_TIMEZONE)
        self._earthlocation = EarthLocation.from_geodetic(lat=self._latitude, lon=self._longitude, height=self._elevation)
        self._observer = Observer(location=self._earthlocation, name=self._observatory, timezone=self._timezone)
        self.status = self.get_status()
        self.condition = 'idle'
    ############ Site info ############
    
    def __repr__(self):
        methods = [f'mainObserver.{name}()\n' for name, method in inspect.getmembers(
            mainObserver, predicate=inspect.isfunction) if not name.startswith('_')]
        txt = '[Methods]\n'+''.join(methods)
        return txt
    
    def get_status(self):
        """
        Returns observation information.
        
        Returns
        =======
        1. obsinfo : dict
            A dictionary containing the following keys and values:
            - update_time: The UTC time at which the observation information was last updated, in ISO format.
            - name_observatory: The name of the observatory.
            - name_observer: The name of the observer.
            - latitude: The latitude of the observatory in degrees.
            - longitude: The longitude of the observatory in degrees.
            - elevation: The elevation of the observatory in meters.
            - timezone: The timezone of the observatory in hours relative to UTC.
            - observer: The astropy.coordinates.EarthLocation object representing the observer's location.
        """
        status = dict()
        now = Time.now()
        status['update_time'] = now.isot
        status['jd'] = round(now.jd,6)
        #status['name_observatory'] = self._observatory
        status['name_observer'] = self._name
        status['latitude'] = round(self._latitude.value,4)
        status['longitude'] = round(self._longitude.value,4)
        status['elevation'] = round(self._elevation.value,2)
        status['moonphase'] = self.moon_phase(now)
        status['timezone'] = self._timezone
        status['observer'] = self._observer
        status['is_connected'] = True
        return status
        
    ############ Time ############
    def localtime(self, 
                  utctimes : Optional[Union[datetime, np.ndarray]] = None):
        """
        Returns the datetime object representing the corresponding local time in the timezone 
        specified by the object's `_timezone` attribute.

        Parameters
        ==========
        1. utctime : datetime, optional
            The datetime object representing the time to convert to local time. If not provided,
            the current UTC time will be used.
            
        Returns
        =======
        1. localtime : datetime
            The datetime object representing the corresponding local time in the timezone 
            specified by the object's `_timezone` attribute.
        """
        
        if utctimes is None:
            utctimes = datetime.utcnow()
        localtime = pytz.utc.localize(utctimes).astimezone(self._timezone)
        localtime = Time(localtime, format="datetime", scale="utc")
        return localtime

    def now(self):
        """
        Get the current UTC time.
        
        Returns
        =======
        1. time : astropy.time.Time
            The current UTC time.
        """
        
        return Time.now()
    
    def is_night(self,
                 utctimes : Optional[Union[datetime, Time, np.ndarray]] = None):
        """
        Check if it is night at a given UTC time and location.
        
        Parameters
        ==========
        1. time : datetime or Time, optional
            The UTC time at which to check if it is night. If not provided, the current time is used.

        Returns
        =======
        1. is_night : bool
            True if it is night at the given time and location, False otherwise.
        """
        
        if utctimes is None:
            utctimes = Time.now()
        if not isinstance(utctimes, Time):
            utctimes = Time(utctimes)
        return self._observer.is_night(utctimes, horizon = -18*u.deg)
    
    def tonight(self,
                time : Optional[Union[datetime, Time, np.ndarray]] = None,
                horizon = -18):
        """
        Get the start and end times of tonight at a given UTC time and location.

        Parameters
        ==========
        1. time : datetime or Time, optional
            The UTC time at which to start the calculation of the start and end of tonight. If not provided, the current time is used.
        2. horizon : float, optional
            The horizon angle to use when calculating the start and end of tonight. Default is -18 degrees.

        Returns
        =======
        1. tonight : tuple
            A tuple of two astropy.time.Time objects representing the start and end times of tonight at the given time and location.
        """

        if time is None:
            time = Time.now()
        if not isinstance(time, Time):
            time = Time(time)
        return self._observer.tonight(time, horizon = horizon*u.deg)
    

    ############ Target ############
    
    def to_altaz(self,
                 ra : Union[float, str, list],
                 dec : Union[float, str, list],
                 utctime : Optional[Union[datetime, Time]] = None):
        """
        Parameters
        ==========
        radec : SkyCoord = Target(RADec) coordinates to convert AltAz
        utctime : datetime.datetime or Time = Time(default = Now)
        
        Return
        ======
        coordinate : SkyCoord = AltAz coordinate of the target
        """
        if utctime is None:
            utctime = Time.now()
        if not isinstance(utctime, Time):
            utctime = Time(utctime)
        coord = self._observer.altaz(utctime, target = self._get_skycoord(ra, dec))
        return coord.altaz
    
    def to_radec(self,
                 alt : float,
                 az : float,
                 utctime : Optional[Union[datetime, Time]] = None):
        """
        Parameters
        ==========
        radec : SkyCoord = Target(AltAz) coordinates to convert RADec
        utctime : datetime.datetime or Time = Time(default = Now)
        
        Return
        ======
        coordinate : SkyCoord = RADec coordinate of the target
        """
        if utctime is None:
            utctime = Time.now()
        if not isinstance(utctime, Time):
            utctime = Time(utctime)
        coord = SkyCoord(alt = alt, az = az, frame ='altaz', location = self._earthlocation, obstime = utctime, unit = 'deg')
        return coord.icrs
    
    def risetime(self,
                 ra : Union[float, str, list],
                 dec : Union[float, str, list],
                 utctime : Optional[Union[datetime, Time]] = None,
                 mode : str = 'next',
                 horizon : float = 30):
        """
        Parameters
        ==========
        radec : SkyCoord = Target(RADec) coordinates to calculate hourangle
        utctime : datetime.datetime or Time = Time(default = Now)
        mode : {‘next’, ‘previous’, ‘nearest’}(default = next)
        horizon : float = horizon angle in degree(default = -18)
        
        Return
        ======
        risetime : Time = Target risetime
        """
        if utctime is None:
            utctime = Time.now()
        if not isinstance(utctime, Time):
            utctime = Time(utctime)
        if not isinstance(self._get_skycoord(ra, dec), SkyCoord):
            raise ValueError('No target is specified')
        rise_time = self._observer.target_rise_time(utctime, self._get_skycoord(ra, dec), which = mode, horizon = horizon*u.deg)
        return rise_time
    
    def settime(self,
                ra : Union[float, str, list],
                dec : Union[float, str, list],
                utctime : Optional[Union[datetime, Time]] = None,
                mode : str = 'nearest',
                horizon : float = 30):
        """
        Parameters
        ==========
        radec : SkyCoord = Target(RADec) coordinates to calculate hourangle
        utctime : datetime.datetime or Time = Time(default = Now)
        mode : {‘next’, ‘previous’, ‘nearest’}(default = nearest)
        horizon : float = horizon angle in degree(default = -18)
        
        Return
        ======
        settime : Time = Target settime
        """
        if utctime is None:
            utctime = Time.now()
        if not isinstance(utctime, Time):
            utctime = Time(utctime)
        if not isinstance(self._get_skycoord(ra, dec), SkyCoord):
            raise ValueError('No target is specified')
        set_time = self._observer.target_set_time(utctime, self._get_skycoord(ra, dec), which = mode, horizon = horizon*u.deg)
        return set_time

    ############ Sun ############
    def sun_radec(self,
                  utctimes : Optional[Union[datetime, Time, np.ndarray]] = None):
        """
        Get the RA and Dec of the Sun at a given UTC time.

        Parameters
        ==========
        1. time : datetime or Time, optional
            The UTC time at which to calculate the RA and Dec of the Sun. If not provided, the current time is used.

        Returns
        =======
        1. sun_radec : astropy.coordinates.SkyCoord
            The RA and Dec of the Sun at the given time, as a SkyCoord object.
        """

        if utctimes is None:
            utctimes = Time.now()
        if not isinstance(utctimes, Time):
            utctimes = Time(utctimes)
        return get_sun(utctimes)
    
    def sun_altaz(self,
                  utctimes : Optional[Union[datetime, Time, np.ndarray]] = None):
        """
        Calculates the altitude and azimuth of the Sun at the given time and location.

        Parameters
        ==========
        1. time : datetime or Time, optional
            The UTC time at which to calculate the altitude and azimuth of the Sun. If not provided, the current time is used.

        Returns
        =======
        1. sun_altaz : astropy.coordinates.AltAz
            The altitude and azimuth of the Sun at the given time and location, as an AltAz object.
        """

        if utctimes is None:
            utctimes = Time.now()
        if not isinstance(utctimes, Time):
            utctimes = Time(utctimes)
        return self._observer.sun_altaz(utctimes)
    
    def sun_risetime(self,
                     utctimes : Optional[Union[datetime, Time, np.ndarray]] = None,
                     mode = 'nearest',
                     horizon = -18):
        """
        Calculates the next rise time of the Sun at the given time and location.

        Parameters
        ==========
        1. time : datetime or Time, optional
            The UTC time at which to calculate the next rise time of the Sun. If not provided, the current time is used.
        2. mode : str, optional
            The method to use for calculating the rise time of the Sun. Can be either 'nearest' (default), 'next', or 'previous'.
        3. horizon : float, optional
            The horizon angle to use when calculating the rise time of the Sun. Default is -18 degrees.

        Returns
        =======
        1. sun_rise_time : astropy.time.Time
            The next rise time of the Sun at the given time and location, as a Time object.
        """

        if utctimes is None:
            utctimes = Time.now()
        if not isinstance(utctimes, Time):
            utctimes = Time(utctimes)
        return self._observer.sun_rise_time(utctimes, which = mode, horizon = horizon * u.deg)
    
    def sun_settime(self,
                    utctimes : Optional[Union[datetime, Time, np.ndarray]] = None,
                    mode = 'nearest',
                    horizon = -18):
        """
        Calculates the next set time of the Sun at the given time and location.

        Parameters
        ==========
        1. time : datetime or Time, optional
            The UTC time at which to calculate the next set time of the Sun. If not provided, the current time is used.
        2. mode : str, optional
            The method to use for calculating the set time of the Sun. Can be either 'nearest' (default), 'next', or 'previous'.
        3. horizon : float, optional
            The horizon angle to use when calculating the set time of the Sun. Default is -18 degrees.

        Returns
        =======
        1. sun_set_time : astropy.time.Time
            The next set time of the Sun at the given time and location, as a Time object.
        """

        if utctimes is None:
            utctimes = Time.now()
        if not isinstance(utctimes, Time):
            utctimes = Time(utctimes)
        return self._observer.sun_set_time(utctimes, which = mode, horizon = horizon * u.deg)
    
    ############ Moon ############
    def moon_radec(self,
                   utctimes : Optional[Union[datetime, Time, np.ndarray]] = None):
        """
        Calculates the RA and Dec of the Moon at the given time and location.
        
        Parameters
        ==========
        1. time : datetime or Time, optional
            The UTC time at which to calculate the RA and Dec of the Moon. If not provided, the current time is used.

        Returns
        =======
        1. moon_radec : astropy.coordinates.SkyCoord
            The RA and Dec of the Moon at the given time, as a SkyCoord object.
        """

        if utctimes == None:
            utctimes = Time.now()
        if not isinstance(utctimes, Time):
            utctimes = Time(utctimes)
        return get_body("moon", utctimes)
    
    def moon_altaz(self,
                   utctimes : Optional[Union[datetime, Time, np.array]] = None):
        """
        Calculates the altitude and azimuth of the Moon at the given time and location.

        Parameters
        ==========
        1. time : datetime or Time, optional
            The UTC time at which to calculate the altitude and azimuth of the Moon. If not provided, the current time is used.

        Returns
        =======
        1. moon_altaz : astropy.coordinates.AltAz
            The altitude and azimuth of the Moon at the given time and location, as an AltAz object.
        """

        if utctimes is None:
            utctimes = Time.now()
        if not isinstance(utctimes, Time):
            utctimes = Time(utctimes)
            
        return self._observer.moon_altaz(utctimes) #self._moon_altaz(radec = get_moon(time), time = time)

    def moon_risetime(self,
                      utctimes : Optional[Union[datetime, Time, np.array]] = None,
                      mode = 'nearest',
                      horizon = -18):
        """
        Calculates the next rise time of the Moon at the given time and location.

        Parameters
        ==========
        1. time : datetime or Time, optional
            The UTC time at which to calculate the next rise time of the Moon. If not provided, the current time is used.
        2. mode : str, optional
            The method to use for calculating the rise time of the Moon. Can be either 'nearest' (default), 'next', or 'previous'.
        3. horizon : float, optional
            The horizon angle to use when calculating the rise time of the Moon. Default is -18 degrees.

        Returns
        =======
        1. moon_rise_time : astropy.time.Time
            The next rise time of the Moon at the given time and location, as a Time object.
        """

        if utctimes is None:
            utctimes = Time.now()
        if not isinstance(utctimes, Time):
            utctimes = Time(utctimes)
        return self._observer.moon_rise_time(utctimes, which = mode, horizon = horizon * u.deg)
    
    def moon_settime(self,
                     utctimes : Optional[Union[datetime, Time, np.array]] = None,
                     mode = 'nearest',
                     horizon = -18):
        """
        Calculates the next set time of the Moon at the given time and location.

        Parameters
        ==========
        1. time : datetime or Time, optional
            The UTC time at which to calculate the next set time of the Moon. If not provided, the current time is used.
        2. mode : str, optional
            The method to use for calculating the set time of the Moon. Can be either 'nearest' (default), 'next', or 'previous'.
        3. horizon : float, optional
            The horizon angle to use when calculating the set time of the Moon. Default is -18 degrees.

        Returns
        =======
        1. moon_set_time : astropy.time.Time
            The next set time of the Moon at the given time and location, as a Time object.
        """

        if utctimes is None:
            utctimes = Time.now()
        if not isinstance(utctimes, Time):
            utctimes = Time(utctimes)
        return self._observer.moon_set_time(utctimes, which = mode, horizon = horizon * u.deg)
    
    def moon_phase(self,
                   utctimes : Optional[Union[datetime, Time, np.array]] = None):
        """
        Calculates the phase of the Moon at the given time and location.
        
        Parameters
        ==========
        1. time : datetime or Time, optional
            The UTC time at which to calculate the phase of the Moon. If not provided, the current time is used.

        Returns
        =======
        1. k : float
            Fraction of moon illuminated
        """
        
        if utctimes is None:
            utctimes = Time.now()
        if not isinstance(utctimes, Time):
            utctimes = Time(utctimes)
        return self._observer.moon_illumination(utctimes)
    
    def _get_skycoord(self,
                     ra : Union[str,float],
                     dec: Union[str,float]):
        """
        Parameters
        ==========
        ra : str | float = Right Ascension, if str, it should be in hms format (e.g., "10:20:30"), if float, it should be in decimal degrees
        dec : str | float = Declination, if str, it should be in dms format (e.g., "+20:30:40"), if float, it should be in decimal degrees
        
        Return
        ======
        coord : SkyCoord = SkyCoord object
        """
        
        # Check if RA and Dec are given as strings (like "10:20:30")
        if isinstance(ra, str) and isinstance(dec, str):
            # Interpret as sexagesimal format (e.g., "10:20:30", "+20:30:40")
            coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
        elif isinstance(ra, (float, int)) and isinstance(dec, (float, int)):
            # Interpret as decimal degrees
            coord = SkyCoord(ra * u.deg, dec * u.deg)
        else:
            raise ValueError("Unsupported RA and Dec format")
        return coord
    
    
# %%
if __name__ == '__main__':
    import astropy.units as u
    obs = mainObserver()
# %%