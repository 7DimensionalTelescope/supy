"""
Star altitude calculations and visibility analysis
Optimized with NumPy vectorization and simplified logic
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, List

from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz
from astropy import units as u
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .observer import Observer


@dataclass
class VisibilityWindow:
    """Data class for an observable window."""
    start_time: Time
    end_time: Time
    max_altitude: float
    min_moon_separation: float
    
    @property
    def duration_hours(self) -> float:
        """Duration of window in hours."""
        return (self.end_time - self.start_time).to(u.hour).value
    
    def contains(self, time: Time) -> bool:
        """Check if time is within this window."""
        return self.start_time <= time <= self.end_time
    
    def time_until_start(self, from_time: Time) -> float:
        """Hours until window starts."""
        if from_time >= self.start_time:
            return 0.0
        return (self.start_time - from_time).to(u.hour).value
    
    def time_remaining(self, from_time: Time) -> float:
        """Hours remaining in window."""
        if from_time >= self.end_time:
            return 0.0
        if from_time < self.start_time:
            return self.duration_hours
        return (self.end_time - from_time).to(u.hour).value


@dataclass
class VisibilityResult:
    """Simplified visibility analysis result."""
    status: str  # "OBSERVABLE" or "NOT_OBSERVABLE"
    when: Optional[str]  # "now", "later", or None
    window: Optional[VisibilityWindow]  # Current or next window
    reason: Optional[str]  # Why not observable
    next_opportunity: Optional[Time]  # Next observable time
    
    @property
    def is_observable(self) -> bool:
        """Check if target is observable."""
        return self.status == "OBSERVABLE"
    
    @property
    def is_observable_now(self) -> bool:
        """Check if target is observable right now."""
        return self.status == "OBSERVABLE" and self.when == "now"


class StarAltitude:
    """
    Optimized star altitude and visibility calculator.
    """
    
    def __init__(self,
                 observer: Optional[Observer] = None):
        """
        Initialize with observer.
        
        Parameters
        ----------
        observer : Observer, optional
            Observer object (default: Chilean observatory)
        """
        self.observer = observer or Observer()
        self._cache = {}  # Cache for expensive calculations
    
    def calculate_visibility(self,
                            ra: float,
                            dec: float,
                            time: Optional[Time] = None,
                            min_altitude: float = 30,
                            min_moon_separation: float = 30,
                            time_resolution: int = 5) -> Tuple[VisibilityResult, dict]:
        """
        Calculate visibility with simplified logic.
        
        Parameters
        ----------
        ra : float
            Right ascension in degrees
        dec : float
            Declination in degrees
        time : Time, optional
            Reference time (default: now)
        min_altitude : float
            Minimum altitude constraint in degrees
        min_moon_separation : float
            Minimum moon separation in degrees
        time_resolution : int
            Time grid resolution in minutes
        
        Returns
        -------
        tuple
            (VisibilityResult, data_dict for plotting)
        """
        if time is None:
            time = self.observer.now()
        
        # Get tonight's window to create a cache key
        # Handle nights where the sun doesn't set below the horizon
        try:
            sunset, sunrise = self.observer.tonight(time)
        except ValueError:
            reason = "Sun does not set below -18 degrees (no astronomical night)"
            result = VisibilityResult("NOT_OBSERVABLE", None, None, reason, None)
            data_dict = {'altitudes': np.array([]), 'moon_separations': np.array([])}
            return result, data_dict
        twilight_times = self.observer.get_twilight_times(time)
        
        # Caching logic
        cache_key = (ra, dec, sunset.jd, min_altitude, min_moon_separation)
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        # Generate time grid for the night
        time_grid = self.observer.get_night_grid(time, time_resolution=time_resolution)
        
        # Calculate all quantities in vectorized operations
        target_altaz = self.observer.target_altaz(ra, dec, time_grid)
        moon_altaz = self.observer.moon_altaz(time_grid)
        sun_altaz = self.observer.sun_altaz(time_grid)
        
        # Extract values as NumPy arrays
        altitudes = target_altaz.alt.deg
        moon_separations = target_altaz.separation(moon_altaz).deg
        sun_altitudes = sun_altaz.alt.deg
        
        # Single-pass visibility calculation
        is_observable = (
            (altitudes > min_altitude) & 
            (moon_separations > min_moon_separation) & 
            (sun_altitudes < -18)  # Astronomical night
        )
        
        # Find continuous observable windows
        windows = self._find_windows(is_observable, time_grid, altitudes, moon_separations)
        
        # Determine visibility status
        result = self._analyze_windows(windows, time, altitudes, moon_separations, 
                                      min_altitude, min_moon_separation)
        
        # Prepare data for plotting
        data_dict = {
            'time_grid': time_grid,
            'altitudes': altitudes,
            'moon_altitudes': moon_altaz.alt.deg,
            'sun_altitudes': sun_altitudes,
            'moon_separations': moon_separations,
            'is_observable': is_observable,
            'sunset': sunset,
            'sunrise': sunrise,
            'twilight_times': twilight_times,  # Include all twilight times
            'min_altitude': min_altitude,
            'min_moon_separation': min_moon_separation,
            'current_time': time,
            'windows': windows,
            'ra': ra,
            'dec': dec
        }
        # Store result in cache before returning
        self._cache[cache_key] = (result, data_dict)
        
        return result, data_dict
    
    def _find_windows(self, is_observable: np.ndarray, time_grid: np.ndarray,
                     altitudes: np.ndarray, moon_separations: np.ndarray) -> List[VisibilityWindow]:
        """
        Find continuous observable windows.
        
        Parameters
        ----------
        is_observable : np.ndarray
            Boolean array of observability
        time_grid : np.ndarray
            Time grid
        altitudes : np.ndarray
            Target altitudes
        moon_separations : np.ndarray
            Moon separations
        
        Returns
        -------
        list
            List of VisibilityWindow objects
        """
        windows = []
        
        # Find transitions in observability
        transitions = np.diff(np.concatenate(([False], is_observable, [False])).astype(int))
        starts = np.where(transitions == 1)[0]
        ends = np.where(transitions == -1)[0] - 1
        
        # Create window objects
        for start_idx, end_idx in zip(starts, ends):
            # Get window data
            window_mask = slice(start_idx, end_idx + 1)
            window_alts = altitudes[window_mask]
            window_moonseps = moon_separations[window_mask]
            
            window = VisibilityWindow(
                start_time=time_grid[start_idx],
                end_time=time_grid[end_idx],
                max_altitude=np.max(window_alts),
                min_moon_separation=np.min(window_moonseps)
            )
            windows.append(window)
        
        return windows
    
    def _analyze_windows(self, windows: List[VisibilityWindow], current_time: Time,
                        altitudes: np.ndarray, moon_separations: np.ndarray,
                        min_altitude: float, min_moon_separation: float) -> VisibilityResult:
        """
        Analyze windows to determine visibility status.
        
        Parameters
        ----------
        windows : list
            List of visibility windows
        current_time : Time
            Current time
        altitudes : np.ndarray
            All altitude values
        moon_separations : np.ndarray
            All moon separation values
        min_altitude : float
            Minimum altitude constraint
        min_moon_separation : float
            Minimum moon separation constraint
        
        Returns
        -------
        VisibilityResult
            Visibility analysis result
        """
        # No windows - not observable
        if not windows:
            # Check if it would have been observable ignoring the sun
            is_potentially_observable = (
                (altitudes > min_altitude) &
                (moon_separations > min_moon_separation)
            )
            if np.any(is_potentially_observable):
                reason = "Target is only observable during daylight/twilight"
            else:
                reason = self._determine_reason(
                    np.max(altitudes) if altitudes.size > 0 else -90,
                    np.min(moon_separations) if moon_separations.size > 0 else 180,
                    min_altitude,
                    min_moon_separation
                )
            
            return VisibilityResult(
                status="NOT_OBSERVABLE",
                when=None,
                window=None,
                reason=reason,
                next_opportunity=None
            )
        
        # Check each window
        for window in windows:
            if window.contains(current_time):
                # Currently observable
                return VisibilityResult(
                    status="OBSERVABLE",
                    when="now",
                    window=window,
                    reason=None,
                    next_opportunity=None
                )
            elif window.start_time > current_time:
                # Observable later
                return VisibilityResult(
                    status="OBSERVABLE",
                    when="later",
                    window=window,
                    reason=None,
                    next_opportunity=window.start_time
                )
        
        # Windows exist but all are in the past
        return VisibilityResult(
            status="NOT_OBSERVABLE",
            when=None,
            window=None,
            reason="Observable window has passed for tonight",
            next_opportunity=None
        )
    
    def _determine_reason(self, max_altitude: float, min_moon_separation: float,
                         min_altitude: float, min_moon_separation_constraint: float) -> str:
        """
        Determine why target is not observable.
        
        Parameters
        ----------
        max_altitude : float
            Maximum altitude achieved
        min_moon_separation : float
            Minimum moon separation achieved
        min_altitude : float
            Required minimum altitude
        min_moon_separation_constraint : float
            Required minimum moon separation
        
        Returns
        -------
        str
            Reason for non-observability
        """
        if max_altitude == -90: # Handle empty altitude array case
             return "Calculation error or no valid time range"
        if max_altitude <= 0:
            return "Target never rises above horizon"
        elif max_altitude < min_altitude and min_moon_separation < min_moon_separation_constraint:
            return f"Both altitude (max {max_altitude:.1f}°) and moon separation (min {min_moon_separation:.1f}°) insufficient"
        elif max_altitude < min_altitude:
            return f"Maximum altitude ({max_altitude:.1f}°) below minimum ({min_altitude}°)"
        elif min_moon_separation < min_moon_separation_constraint:
            return f"Moon too close (minimum separation {min_moon_separation:.1f}°)"
        else:
            return "Timing constraints not met"
    
    def find_next_observable_night(self,
                                   ra: float,
                                   dec: float,
                                   time: Optional[Time] = None,
                                   min_altitude: float = 30,
                                   min_moon_separation: float = 30,
                                   search_days: int = 7) -> Optional[Tuple[VisibilityResult, dict, int]]:
        """
        Find the next observable night within a given search window.

        Parameters
        ----------
        ra : float
            Right ascension in degrees
        dec : float
            Declination in degrees
        time : Time, optional
            Start time for the search (default: now)
        min_altitude : float
            Minimum altitude constraint
        min_moon_separation : float
            Minimum moon separation constraint
        search_days : int
            Number of days to search ahead

        Returns
        -------
        tuple or None
            (VisibilityResult, data_dict, days_from_now) if an opportunity is found,
            otherwise None.
        """
        if time is None:
            time = self.observer.now()

        for days_ahead in range(1, search_days + 1):
            future_time = time + days_ahead * u.day
            
            result, data = self.calculate_visibility(
                ra, dec, time=future_time,
                min_altitude=min_altitude,
                min_moon_separation=min_moon_separation
            )
            
            if result.is_observable:
                return result, data, days_ahead
        
        return None
    
    def plot_visibility(self, data_dict: dict, result: VisibilityResult, 
                       target_name: str = "Target", show_current: bool = True) -> plt.Figure:
        """
        Create optimized visibility plot.
        
        Parameters
        ----------
        data_dict : dict
            Data from calculate_visibility
        result : VisibilityResult
            Visibility analysis result
        target_name : str
            Name for plot title
        show_current : bool
            Whether to show current time marker
        
        Returns
        -------
        Figure
            Matplotlib figure
        """
        # Extract data
        time_grid = data_dict['time_grid']
        altitudes = data_dict['altitudes']
        moon_altitudes = data_dict['moon_altitudes']
        is_observable = data_dict['is_observable']
        sunset = data_dict['sunset']
        sunrise = data_dict['sunrise']
        twilight_times = data_dict.get('twilight_times', {})
        min_altitude = data_dict['min_altitude']
        current_time = data_dict['current_time']
        
        # Convert times for plotting
        time_plot = time_grid.plot_date
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
        
        # Plot altitude curves
        ax.scatter(time_plot[is_observable], altitudes[is_observable], 
                  c='green', s=20, alpha=0.6, label='Observable')
        ax.scatter(time_plot[~is_observable], altitudes[~is_observable], 
                  c='red', s=20, alpha=0.3, label='Not observable')
        
        # Moon altitude
        ax.plot(time_plot, moon_altitudes, 'b-', alpha=0.5, linewidth=1, label='Moon')
        
        # Different twilight shadings (if available)
        if twilight_times:
            # Civil twilight (lightest)
            if 'sunset_civil' in twilight_times and 'sunrise_civil' in twilight_times:
                civil_mask = (time_grid >= twilight_times['sunset_civil']) & \
                           (time_grid <= twilight_times['sunrise_civil'])
                if np.any(civil_mask):
                    ax.fill_between(time_plot, 0, 90, where=civil_mask, 
                                  alpha=0.05, color='blue', label='Civil twilight')
            
            # Nautical twilight
            if 'sunset_nautical' in twilight_times and 'sunrise_nautical' in twilight_times:
                nautical_mask = (time_grid >= twilight_times['sunset_nautical']) & \
                              (time_grid <= twilight_times['sunrise_nautical'])
                if np.any(nautical_mask):
                    ax.fill_between(time_plot, 0, 90, where=nautical_mask, 
                                  alpha=0.1, color='blue')
            
            # Astronomical twilight
            if 'sunset_astro' in twilight_times and 'sunrise_astro' in twilight_times:
                astro_mask = (time_grid >= twilight_times['sunset_astro']) & \
                            (time_grid <= twilight_times['sunrise_astro'])
                if np.any(astro_mask):
                    ax.fill_between(time_plot, 0, 90, where=astro_mask, 
                                  alpha=0.15, color='navy')
        
        # Astronomical night (darkest)
        night_mask = (time_grid >= sunset) & (time_grid <= sunrise)
        if np.any(night_mask):
            ax.fill_between(time_plot, 0, 90, where=night_mask, 
                          alpha=0.2, color='navy', label='Astronomical night')
        
        # Minimum altitude line
        ax.axhline(y=min_altitude, color='orange', linestyle='--', 
                  alpha=0.7, label=f'Min altitude ({min_altitude}°)')
        
        # Current time marker
        if show_current and (result.is_observable_now or result.when == "later"):
            ax.axvline(x=current_time.plot_date, color='purple', 
                      linestyle='-', linewidth=2, label='Now')
        
        # Observable windows
        for window in data_dict.get('windows', []):
            ax.axvspan(window.start_time.plot_date, window.end_time.plot_date,
                      alpha=0.2, color='green')
        
        # Formatting
        ax.set_xlabel('Time (UTC)')
        ax.set_ylabel('Altitude (degrees)')
        ax.set_title(f'{target_name} Visibility - {result.status}')
        ax.set_ylim(0, 90)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add visibility info text
        if result.is_observable:
            info_text = self._format_info_text(result)
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def _format_info_text(self, result: VisibilityResult) -> str:
        """Format information text for plot."""
        lines = []
        
        if result.when == "now":
            remaining = result.window.time_remaining(Time.now())
            lines.append(f"Currently Observable")
            lines.append(f"Remaining: {remaining:.1f} hours")
            lines.append(f"Until: {result.window.end_time.datetime.strftime('%H:%M UTC')}")
        elif result.when == "later":
            wait_time = result.window.time_until_start(Time.now())
            lines.append(f"Observable in {wait_time:.1f} hours")
            lines.append(f"Duration: {result.window.duration_hours:.1f} hours")
            lines.append(f"Window: {result.window.start_time.datetime.strftime('%H:%M')}-"
                        f"{result.window.end_time.datetime.strftime('%H:%M')} UTC")
        
        if result.window:
            lines.append(f"Max altitude: {result.window.max_altitude:.1f}°")
        
        return '\n'.join(lines)
