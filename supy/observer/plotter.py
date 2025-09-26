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
        Comprehensive visibility analysis for tonight and future opportunities.

        Returns
        -------
        dict
            A standardized dictionary with comprehensive visibility analysis.
        """
        # 1. Analyze tonight's visibility
        tonight_result, tonight_data = self.staralt.calculate_visibility(
            ra, dec,
            min_altitude=min_altitude,
            min_moon_separation=min_moon_separation
        )

        analysis = {
            "target_info": {"name": target_name, "ra": ra, "dec": dec},
            "tonight": self._format_night_result(tonight_result, Time.now()),
            "next_opportunity": None,
            "summary": {},
            "plot_data_for": "tonight",
            "plot_data": tonight_data
        }

        # 2. If not observable tonight, search for the next opportunity
        if not tonight_result.is_observable:
            next_opp_result = self.staralt.find_next_observable_night(
                ra, dec,
                min_altitude=min_altitude,
                min_moon_separation=min_moon_separation,
                search_days=search_days
            )
            if next_opp_result:
                result, data, days_from_now = next_opp_result
                analysis["next_opportunity"] = self._format_night_result(result, Time.now(), days_from_now)
                analysis["plot_data_for"] = "next_opportunity"
                analysis["plot_data"] = data
        
        # 3. Generate summary
        analysis["summary"] = self._generate_summary(analysis)

        return analysis

    def _format_night_result(self, result: VisibilityResult, time_now: Time, days_from_now: int = 0) -> dict:
        """Helper to format a VisibilityResult into a dictionary."""
        night_info = {
            "date": (time_now + days_from_now * u.day).datetime.strftime('%Y-%m-%d'),
            "status": result.status,
            "when": result.when,
            "reason": result.reason,
            "window": None
        }
        if days_from_now > 0:
            night_info["days_from_now"] = days_from_now

        if result.window:
            night_info["window"] = {
                "start_time_utc": result.window.start_time.isot,
                "end_time_utc": result.window.end_time.isot,
                "duration_hours": result.window.duration_hours,
                "max_altitude": result.window.max_altitude,
            }
            if result.when == "now":
                night_info["window"]["time_remaining_hours"] = result.window.time_remaining(time_now)
            elif result.when == "later":
                night_info["window"]["time_until_start_hours"] = result.window.time_until_start(time_now)
        
        return night_info

    def _generate_summary(self, analysis: dict) -> dict:
        """Helper to generate the summary block."""
        is_observable_tonight = analysis["tonight"]["status"] == "OBSERVABLE"
        recommendation = "Consider different target or relaxed constraints"
        
        if is_observable_tonight:
            when = analysis["tonight"]["when"]
            window = analysis["tonight"]["window"]
            if when == "now":
                remaining = window.get("time_remaining_hours", 0)
                if remaining < 1: recommendation = "URGENT: Observe immediately!"
                else: recommendation = "Good conditions for observation"
            elif when == "later":
                wait_time = window.get("time_until_start_hours", 0)
                recommendation = f"Prepare equipment - observations in {wait_time:.1f} hours"
        elif analysis["next_opportunity"]:
            days = analysis["next_opportunity"]["days_from_now"]
            recommendation = f"Target is observable in {days} days"
        else:
            recommendation = "No observable window found for the searched period"

        summary = {
            "is_observable_tonight": is_observable_tonight,
            "recommendation": recommendation,
            "formatted_message": self.format_message(analysis) # format_message needs to be called last
        }
        return summary
    
    def create_plot(self,
                   analysis: Dict[str, Any],
                   save_path: Optional[str] = None) -> Optional[str]:
        """
        Create visibility plot from a pre-computed analysis object.
        """
        plot_data = analysis.get("plot_data")
        if not plot_data or not plot_data.get('altitudes').any():
            return None # Cannot plot if there's no data

        target_name = analysis["target_info"]["name"]
        
        # Determine which night's data we are plotting
        plot_source = analysis["plot_data_for"]
        night_info = analysis[plot_source]
        result = self.staralt.calculate_visibility( # We need the result object for the plot function
            analysis['target_info']['ra'],
            analysis['target_info']['dec'],
            time=Time(night_info['date'])
        )[0]


        plot_title = f"{target_name} on {night_info['date']}"
        show_current = (plot_source == "tonight")

        fig = self.staralt.plot_visibility(plot_data, result, plot_title, show_current)

        if save_path is None:
            fd, save_path = tempfile.mkstemp(suffix='.png')
            os.close(fd)
        
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return save_path
    
    def format_message(self, analysis: Dict[str, Any]) -> str:
        """
        Format visibility analysis as human-readable message from the analysis object.
        """
        lines = []
        tonight = analysis["tonight"]
        
        # Tonight's Status
        if tonight["status"] == "OBSERVABLE":
            if tonight["when"] == "now":
                lines.append("🟢 *CURRENTLY OBSERVABLE*")
                remaining = tonight["window"].get("time_remaining_hours", 0)
                lines.append(f"⏱️ *Remaining*: {remaining:.1f} hours")
            else: # "later"
                lines.append("🟡 *OBSERVABLE LATER TONIGHT*")
                wait_time = tonight["window"].get("time_until_start_hours", 0)
                lines.append(f"🕐 *Starts in*: {wait_time:.1f} hours")
            
            # Add window details for tonight if observable
            window = tonight["window"]
            if window:
                lines.append(f"📊 *Max altitude*: {window['max_altitude']:.1f}°")
                lines.append(f"⏳ *Duration*: {window['duration_hours']:.1f} hours")

        else: # Not observable tonight
            next_opp = analysis["next_opportunity"]
            if next_opp:
                # Not observable tonight but will be observable later
                lines.append("🟠 *NOT OBSERVABLE TONIGHT*")
                lines.append(f"❓ *Reason*: {tonight['reason']}")
                lines.append(f"📅 *Next opportunity*: {next_opp['date']} "
                             f"(in {next_opp['days_from_now']} days)")
            else:   
                # Not observable tonight and no next opportunity
                lines.append("❌ *NOT OBSERVABLE*")
                lines.append(f"❓ *Reason for tonight*: {tonight['reason']}")

        # Add overall recommendation
        if analysis.get("summary"):
            lines.append(f"💡 {analysis['summary']['recommendation']}")
        
        return '\n'.join(lines)
    
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
