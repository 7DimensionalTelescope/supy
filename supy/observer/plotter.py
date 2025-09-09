"""
High-level visibility plotter for easy use
"""

import os
import tempfile
from typing import Optional, Tuple, Dict, Any
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
from astropy.time import Time
from astropy import units as u

from .observer import Observer
from .staralt import StarAltitude, VisibilityResult


class VisibilityPlotter:
    """
    Visibility plotter for astronomical observations.
    """
    
    def __init__(self, observer: Optional[Observer] = None):
        """
        Initialize plotter.
        
        Parameters
        ----------
        observer : Observer, optional
            Observer object (default: Chilean observatory)
        """
        self.observer = observer or Observer()
        self.staralt = StarAltitude(self.observer)
    
    def analyze_visibility(self, 
                          ra: float, 
                          dec: float,
                          target_name: str = "Target",
                          min_altitude: float = 30,
                          min_moon_separation: float = 30,
                          search_days: int = 7) -> Dict[str, Any]:
        """
        Comprehensive visibility analysis.
        
        Parameters
        ----------
        ra : float
            Right ascension in degrees
        dec : float
            Declination in degrees
        target_name : str
            Target name for display
        min_altitude : float
            Minimum altitude constraint in degrees
        min_moon_separation : float
            Minimum moon separation in degrees
        search_days : int
            Number of days to search for next opportunity
        
        Returns
        -------
        dict
            Visibility analysis results
        """
        # Analyze current/tonight's visibility
        result, data = self.staralt.calculate_visibility(
            ra, dec,
            min_altitude=min_altitude,
            min_moon_separation=min_moon_separation
        )
        
        # If not observable tonight, search future nights
        next_observable = None
        if not result.is_observable:
            next_observable = self._find_next_observable(
                ra, dec, min_altitude, min_moon_separation, search_days
            )
        
        # Build comprehensive result
        analysis = {
            'target_name': target_name,
            'ra': ra,
            'dec': dec,
            'current_status': result.status,
            'when_observable': result.when,
            'current_window': None,
            'next_opportunity': None,
            'recommendation': self._get_recommendation(result, next_observable),
            'urgency': self._get_urgency(result),
            'plot_data': data
        }
        
        # Add window details if available
        if result.window:
            analysis['current_window'] = {
                'start': result.window.start_time.datetime,
                'end': result.window.end_time.datetime,
                'duration_hours': result.window.duration_hours,
                'max_altitude': result.window.max_altitude,
                'remaining_hours': result.window.time_remaining(Time.now()) if result.when == "now" else None
            }
        
        # Add next opportunity if found
        if next_observable:
            analysis['next_opportunity'] = {
                'date': next_observable['date'],
                'window_start': next_observable['window_start'],
                'window_duration': next_observable['duration_hours'],
                'days_from_now': next_observable['days_from_now']
            }
        
        return analysis
    
    def create_plot(self,
                   ra: float,
                   dec: float,
                   target_name: str = "Target",
                   min_altitude: float = 30,
                   min_moon_separation: float = 30,
                   save_path: Optional[str] = None) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Create visibility plot.
        
        Parameters
        ----------
        ra : float
            Right ascension in degrees
        dec : float
            Declination in degrees
        target_name : str
            Target name for plot title
        min_altitude : float
            Minimum altitude constraint
        min_moon_separation : float
            Minimum moon separation constraint
        save_path : str, optional
            Path to save plot (default: temporary file)
        
        Returns
        -------
        tuple
            (plot_path, visibility_analysis)
        """
        # Get visibility analysis
        analysis = self.analyze_visibility(
            ra, dec, target_name, min_altitude, min_moon_separation
        )
        
        # Skip plot if not observable
        if analysis['current_status'] == "NOT_OBSERVABLE" and not analysis.get('next_opportunity'):
            return None, analysis
        
        # Determine which night to plot
        if analysis['current_status'] == "OBSERVABLE":
            # Plot tonight
            result, data = self.staralt.calculate_visibility(
                ra, dec, min_altitude=min_altitude, min_moon_separation=min_moon_separation
            )
            plot_title = target_name
            show_current = True
        else:
            # Plot next opportunity
            next_opp = analysis.get('next_opportunity')
            if next_opp:
                # Calculate for the next observable night
                future_time = Time.now() + next_opp['days_from_now'] * u.day
                result, data = self.staralt.calculate_visibility(
                    ra, dec, time=future_time,
                    min_altitude=min_altitude, min_moon_separation=min_moon_separation
                )
                plot_title = f"{target_name} - {next_opp['date']}"
                show_current = False
            else:
                return None, analysis
        
        # Create plot
        fig = self.staralt.plot_visibility(data, result, plot_title, show_current)
        
        # Save plot
        if save_path is None:
            fd, save_path = tempfile.mkstemp(suffix='.png')
            os.close(fd)
        
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return save_path, analysis
    
    def format_message(self, analysis: Dict[str, Any]) -> str:
        """
        Format visibility analysis as human-readable message.
        
        Parameters
        ----------
        analysis : dict
            Visibility analysis results
        
        Returns
        -------
        str
            Formatted message
        """
        lines = []
        
        # Header
        status = analysis['current_status']
        if status == "OBSERVABLE":
            if analysis['when_observable'] == "now":
                lines.append("🟢 **CURRENTLY OBSERVABLE**")
                if analysis['current_window']:
                    remaining = analysis['current_window'].get('remaining_hours', 0)
                    lines.append(f"⏱️ Remaining: {remaining:.1f} hours")
            else:
                lines.append("🟡 **OBSERVABLE LATER TONIGHT**")
                if analysis['current_window']:
                    start = analysis['current_window']['start']
                    lines.append(f"🕐 Starts at: {start.strftime('%H:%M UTC')}")
        else:
            lines.append("🔴 **NOT OBSERVABLE TONIGHT**")
        
        # Add window details
        if analysis.get('current_window'):
            window = analysis['current_window']
            lines.append(f"📊 Max altitude: {window['max_altitude']:.1f}°")
            lines.append(f"⏳ Duration: {window['duration_hours']:.1f} hours")
        
        # Add next opportunity
        if analysis.get('next_opportunity'):
            next_opp = analysis['next_opportunity']
            lines.append(f"📅 Next opportunity: {next_opp['date']} "
                        f"(in {next_opp['days_from_now']} days)")
        
        # Add recommendation
        lines.append(f"💡 {analysis['recommendation']}")
        
        return '\n'.join(lines)
    
    def _find_next_observable(self, ra: float, dec: float, 
                             min_altitude: float, min_moon_separation: float,
                             search_days: int) -> Optional[Dict[str, Any]]:
        """
        Find next observable opportunity.
        
        Parameters
        ----------
        ra : float
            Right ascension in degrees
        dec : float
            Declination in degrees
        min_altitude : float
            Minimum altitude constraint
        min_moon_separation : float
            Minimum moon separation constraint
        search_days : int
            Days to search ahead
        
        Returns
        -------
        dict or None
            Next opportunity details
        """
        current_time = Time.now()
        
        for days_ahead in range(1, search_days + 1):
            future_time = current_time + days_ahead * u.day
            
            result, _ = self.staralt.calculate_visibility(
                ra, dec, time=future_time,
                min_altitude=min_altitude,
                min_moon_separation=min_moon_separation
            )
            
            if result.is_observable:
                return {
                    'date': future_time.datetime.strftime('%Y-%m-%d'),
                    'window_start': result.window.start_time.datetime,
                    'duration_hours': result.window.duration_hours,
                    'days_from_now': days_ahead
                }
        
        return None
    
    def _get_recommendation(self, result: VisibilityResult, 
                           next_opportunity: Optional[Dict]) -> str:
        """
        Get observation recommendation.
        
        Parameters
        ----------
        result : VisibilityResult
            Current visibility result
        next_opportunity : dict or None
            Next opportunity details
        
        Returns
        -------
        str
            Recommendation text
        """
        if result.when == "now":
            remaining = result.window.time_remaining(Time.now())
            if remaining < 0.5:
                return "URGENT: Begin observations immediately!"
            elif remaining < 2:
                return "Begin observations soon"
            else:
                return "Good conditions for observation"
        elif result.when == "later":
            wait_time = result.window.time_until_start(Time.now())
            if wait_time < 1:
                return "Prepare for observations - starting soon"
            else:
                return f"Prepare equipment - observations in {wait_time:.1f} hours"
        elif next_opportunity:
            days = next_opportunity['days_from_now']
            return f"Target will be observable in {days} days"
        else:
            return "Consider different target or relaxed constraints"
    
    def _get_urgency(self, result: VisibilityResult) -> str:
        """
        Determine urgency level.
        
        Parameters
        ----------
        result : VisibilityResult
            Visibility result
        
        Returns
        -------
        str
            Urgency level: 'critical', 'high', 'medium', 'low', 'none'
        """
        if not result.is_observable:
            return 'none'
        
        if result.when == "now":
            remaining = result.window.time_remaining(Time.now())
            if remaining < 0.5:
                return 'critical'
            elif remaining < 1:
                return 'high'
            elif remaining < 2:
                return 'medium'
            else:
                return 'low'
        elif result.when == "later":
            wait_time = result.window.time_until_start(Time.now())
            if wait_time < 0.5:
                return 'high'
            elif wait_time < 1:
                return 'medium'
            else:
                return 'low'
        
        return 'low'


# Convenience function for quick checks
def quick_check(ra: float, dec: float, target_name: str = "Target") -> None:
    """
    Quick visibility check with printed results.
    
    Parameters
    ----------
    ra : float
        Right ascension in degrees
    dec : float
        Declination in degrees
    target_name : str
        Target name
    """
    plotter = VisibilityPlotter()
    analysis = plotter.analyze_visibility(ra, dec, target_name)
    message = plotter.format_message(analysis)
    print(message)
    
    # Create plot if observable
    if analysis['current_status'] == "OBSERVABLE" or analysis.get('next_opportunity'):
        plot_path, _ = plotter.create_plot(ra, dec, target_name)
        if plot_path:
            print(f"\nPlot saved to: {plot_path}")
