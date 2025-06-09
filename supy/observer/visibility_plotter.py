import os
import tempfile
import time
from datetime import datetime, timedelta
from .mainobserver import mainObserver
from .staralt import Staralt
import matplotlib.pyplot as plt
import logging
import pytz
from typing import Optional, Dict, Tuple, Any, List, Union

# Logger only for standalone testing
test_logger = logging.getLogger(__name__)

class VisibilityPlotter:
    """
    Handle visibility plot generation and Slack uploading for GCN notices.
    
    This class creates visibility plots using staralt.py and handles 
    temporary file management for uploading to Slack. It provides specialized
    visibility analysis for GRB observations from Chile.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the observer and plotter.
        
        Args:
            logger: Logger instance from main application
        """
        self.observer = mainObserver()  # Use default parameters
        self.staralt = Staralt(self.observer)
        self.logger = logger if logger else test_logger
        
        # Define timezones
        self.chile_tz = pytz.timezone("America/Santiago")
        self.korea_tz = pytz.timezone("Asia/Seoul")
    
    def _convert_time_to_clt_kst(self, utc_time: datetime) -> Tuple[datetime, datetime]:
        """
        Convert UTC time to Chile local time and Korean time.
        
        Args:
            utc_time: Datetime in UTC
            
        Returns:
            Tuple containing (chile_time, korea_time)
        """
        # Ensure UTC time has timezone info
        if utc_time.tzinfo is None:
            utc_time = pytz.utc.localize(utc_time)
            
        # Convert to Chile and Korea times
        chile_time = utc_time.astimezone(self.chile_tz)
        korea_time = utc_time.astimezone(self.korea_tz)
        
        return chile_time, korea_time
    
    def _format_time_clt_kst(self, utc_time: Optional[datetime]) -> str:
        """
        Format time in both CLT and KST timezones.
        
        Args:
            utc_time: Datetime in UTC, or None
            
        Returns:
            String with formatted time in both timezones, or "Unknown" if utc_time is None
        """
        if utc_time is None:
            return "Unknown"
            
        chile_time, korea_time = self._convert_time_to_clt_kst(utc_time)
        return f"{chile_time.strftime('%H:%M')} CLT / {korea_time.strftime('%H:%M')} KST"
    
    def _analyze_visibility_status(self, staralt_data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze visibility data and determine one of 4 clear statuses:
        1. observable_now: Currently observable
        2. observable_later: Observable later tonight  
        3. observable_tomorrow: Not observable tonight, but will be tomorrow night
        4. not_observable: Not observable at all
        """
        result = {
            "status": "not_observable",
            "condition": "Unknown",
            "observable_hours": 0,
            "observable_start": None,
            "observable_end": None,
            "best_time": None,
            "current_altitude": 0,
            "current_moon_separation": 0,
            "reason": "Unknown limitation",
            "message": "Not observable",
            "recommendation": "Observation not possible"
        }
        
        try:
            # Extract data from staralt analysis
            now_datetime = staralt_data_dict.get("now_datetime")
            color_target = staralt_data_dict.get("color_target", [])
            target_times = staralt_data_dict.get("target_times", [])
            target_alts = staralt_data_dict.get("target_alts", [])
            target_moonsep = staralt_data_dict.get("target_moonsep", [])
            min_altitude = staralt_data_dict.get("target_minalt", 30)
            min_moon_sep = staralt_data_dict.get("target_minmoonsep", 30)
            tonight = staralt_data_dict.get("tonight", {})
            
            # Validate required data
            if not all([now_datetime, color_target, target_times, target_alts, target_moonsep]):
                result["reason"] = "Insufficient data for visibility analysis"
                return result
            
            # Convert target_times to datetime objects
            target_times_dt = []
            for time_str in target_times:
                if isinstance(time_str, str):
                    target_times_dt.append(datetime.fromisoformat(time_str))
                else:
                    target_times_dt.append(time_str)
            
            # Find current position in time array
            now_idx = min(range(len(target_times_dt)), 
                          key=lambda i: abs((target_times_dt[i] - now_datetime).total_seconds()))
            
            # Get current conditions
            result["current_altitude"] = target_alts[now_idx]
            result["current_moon_separation"] = target_moonsep[now_idx]
            
            # Find observable periods (green points)
            observable_indices = [i for i, color in enumerate(color_target) if color == 'g']
            
            # Case 1: Currently observable
            if observable_indices and now_idx in observable_indices:
                return self._handle_observable_now(result, observable_indices, target_times_dt, now_datetime)
            
            # Case 2: Observable later tonight
            elif observable_indices and any(i > now_idx for i in observable_indices):
                return self._handle_observable_later(result, observable_indices, target_times_dt, now_datetime, now_idx)
            
            # Case 3 & 4: Not observable tonight - check if tomorrow is possible
            else:
                return self._handle_not_observable_tonight(result, target_alts, target_moonsep, min_altitude, min_moon_sep, tonight, now_datetime)
                
        except Exception as e:
            self.logger.error(f"Error analyzing visibility: {e}", exc_info=True)
            result["reason"] = f"Error in visibility analysis: {str(e)}"
            return result
    
    def _handle_observable_now(self, result: Dict[str, Any], observable_indices: List[int], 
                              target_times_dt: List[datetime], now_datetime: datetime) -> Dict[str, Any]:
        """Handle Case 1: Currently observable"""
        result["status"] = "observable_now"
        
        # Find the current observable window
        start_idx = min(observable_indices)
        end_idx = max(observable_indices)
        
        result["observable_start"] = target_times_dt[start_idx]
        result["observable_end"] = target_times_dt[end_idx]
        result["observable_hours"] = (result["observable_end"] - result["observable_start"]).total_seconds() / 3600
        
        # Calculate remaining time
        remaining_seconds = (result["observable_end"] - now_datetime).total_seconds()
        result["remaining_hours"] = max(0, remaining_seconds / 3600)
        
        # Set condition based on current altitude and remaining time
        if result["current_altitude"] > 60 and result["remaining_hours"] > 2:
            result["condition"] = "Excellent Observing Conditions"
        elif result["current_altitude"] > 45 and result["remaining_hours"] > 1:
            result["condition"] = "Good Observing Conditions"
        elif result["remaining_hours"] < 1:
            result["condition"] = "Limited Time Remaining"
        else:
            result["condition"] = "Acceptable Observing Conditions"
        
        result["recommendation"] = "Begin observations immediately"
        return result
    
    def _handle_observable_later(self, result: Dict[str, Any], observable_indices: List[int], 
                                target_times_dt: List[datetime], now_datetime: datetime, now_idx: int) -> Dict[str, Any]:
        """Handle Case 2: Observable later tonight"""
        result["status"] = "observable_later"
        
        # Find next observable window
        future_indices = [i for i in observable_indices if i > now_idx]
        start_idx = min(future_indices)
        end_idx = max(observable_indices)
        
        result["observable_start"] = target_times_dt[start_idx]
        result["observable_end"] = target_times_dt[end_idx]
        result["observable_hours"] = (result["observable_end"] - result["observable_start"]).total_seconds() / 3600
        
        # Calculate time until observable
        hours_until = (result["observable_start"] - now_datetime).total_seconds() / 3600
        result["hours_until_observable"] = hours_until
        
        # Set condition based on wait time
        if hours_until < 1:
            result["condition"] = "Observable Very Soon"
        elif hours_until < 3:
            result["condition"] = "Observable in a Few Hours"
        else:
            result["condition"] = "Long Wait for Observation"
        
        result["recommendation"] = f"Schedule observations to begin at {self._format_time_clt_kst(result['observable_start'])}"
        return result
    
    def _handle_not_observable_tonight(self, result: Dict[str, Any], target_alts: List[float], 
                                      target_moonsep: List[float], min_altitude: float, min_moon_sep: float,
                                      tonight: Dict[str, Any], now_datetime: datetime) -> Dict[str, Any]:
        """Handle Cases 3 & 4: Not observable tonight"""
        
        # Get tonight's observing window times
        sunset_night = tonight.get("sunset_night")
        sunrise_night = tonight.get("sunrise_night")
        
        # Check if we're past tonight's observing window
        is_past_observing_window = False
        if sunset_night and sunrise_night and now_datetime:
            # If current time is past sunrise (astronomical dawn), tonight's window is over
            if now_datetime > sunrise_night:
                is_past_observing_window = True
        
        # Get maximum conditions tonight
        max_alt = max(target_alts) if target_alts else 0
        min_moonsep_tonight = min(target_moonsep) if target_moonsep else 0
        
        # Determine if target might be observable tomorrow
        # Case 3: Observable tomorrow - target has reasonable altitude potential
        might_be_observable_tomorrow = False
        tomorrow_reason = ""
        
        # Check altitude potential
        if max_alt > min_altitude - 20:  # Within 20 degrees of minimum
            might_be_observable_tomorrow = True
            if max_alt < min_altitude:
                tomorrow_reason = f"Target reaches {max_alt:.1f}° tonight (close to {min_altitude}° minimum)"
            else:
                tomorrow_reason = f"Target reaches good altitude ({max_alt:.1f}°) but timing/moon issues tonight"
        
        # Check if moon separation was the main issue
        if max_alt >= min_altitude and min_moonsep_tonight < min_moon_sep:
            might_be_observable_tomorrow = True
            tomorrow_reason = f"Good altitude ({max_alt:.1f}°) but moon too close ({min_moonsep_tonight:.1f}°)"
        
        # Special case: if we're past tonight's observing window and target had potential
        if is_past_observing_window and max_alt > min_altitude - 30:
            might_be_observable_tomorrow = True
            tomorrow_reason = "Tonight's observing window has ended, check tomorrow"
        
        # Case 3: Observable tomorrow
        if might_be_observable_tomorrow:
            result["status"] = "observable_tomorrow"
            result["condition"] = "Likely Observable Tomorrow"
            result["reason"] = tomorrow_reason
            result["recommendation"] = "Check visibility for tomorrow night"
            return result
        
        # Case 4: Not observable at all
        result["status"] = "not_observable"
        
        if max_alt <= 0:
            result["condition"] = "Never Rises"
            result["reason"] = "Target never rises above horizon from Chile"
        elif max_alt < min_altitude - 20:
            result["condition"] = "Too Low Altitude"
            result["reason"] = f"Target maximum altitude ({max_alt:.1f}°) far below minimum ({min_altitude}°)"
        elif min_moonsep_tonight < min_moon_sep - 20:
            result["condition"] = "Severe Moon Interference"
            result["reason"] = f"Target too close to Moon (min separation: {min_moonsep_tonight:.1f}°)"
        else:
            result["condition"] = "Multiple Limitations"
            result["reason"] = f"Altitude ({max_alt:.1f}°) and moon separation ({min_moonsep_tonight:.1f}°) issues"
        
        result["recommendation"] = "Observation not feasible from Chile"
        return result

    def _generate_today_visibility(self, ra: float, dec: float, grb_name: str, minalt: float, minmoonsep: float) -> Tuple[Dict[str, Any], Any]:
        """Generate visibility data for today/tonight"""
        self.staralt.set_target(
            ra=ra,
            dec=dec,
            objname=grb_name,
            target_minalt=minalt,
            target_minmoonsep=minmoonsep
        )
        
        visibility_info = self._analyze_visibility_status(self.staralt.data_dict)
        return visibility_info, self.staralt.data_dict

    def _generate_tomorrow_visibility(self, ra: float, dec: float, grb_name: str, minalt: float, minmoonsep: float) -> Tuple[Dict[str, Any], Any]:
        """Generate visibility data for tomorrow night"""
        # Calculate tomorrow's date (next observing night)
        tomorrow = datetime.now() + timedelta(days=1)
        
        self.staralt.set_target(
            ra=ra,
            dec=dec,
            objname=grb_name,
            utctime=tomorrow,
            target_minalt=minalt,
            target_minmoonsep=minmoonsep
        )
        
        tomorrow_visibility = self._analyze_visibility_status(self.staralt.data_dict)
        return tomorrow_visibility, self.staralt.data_dict

    def create_visibility_plot(self, ra, dec, grb_name=None, test_mode=False, minalt=30, minmoonsep=30, savefig=True):
        """
        Create visibility plot based on 4 clear scenarios:
        1. observable_now: Show today's plot with current time marker
        2. observable_later: Show today's full night plot with current time marker  
        3. observable_tomorrow: Show tomorrow's plot with warning label (no current time)
        4. not_observable: Return no plot
        
        Args:
            ra: Right Ascension in degrees
            dec: Declination in degrees
            grb_name: Name of the GRB for plot title
            test_mode: If True, save plot to test_plots directory
            minalt: Minimum altitude for target in degrees
            minmoonsep: Minimum moon separation for target in degrees
            savefig: Whether to save the figure
            
        Returns:
            tuple: (plot_path, visibility_info) or (None, visibility_info)
        """
        try:
            # Step 1: Analyze today's visibility
            today_visibility, today_data = self._generate_today_visibility(ra, dec, grb_name or "Target", minalt, minmoonsep)
            
            status = today_visibility.get("status")
            self.logger.info(f"Visibility status: {status}")
            
            # Step 2: Handle each case
            if status == "not_observable":
                # Case 4: No plot
                self.logger.info("Target not observable - no plot generated")
                return None, today_visibility
            
            elif status == "observable_tomorrow":
                # Case 3: Generate tomorrow's plot
                self.logger.info("Generating tomorrow's visibility plot")
                tomorrow_visibility, tomorrow_data = self._generate_tomorrow_visibility(ra, dec, grb_name or "Target", minalt, minmoonsep)
                
                # Use tomorrow's data for plotting but keep today's status info
                plot_data = tomorrow_data
                final_visibility = today_visibility.copy()
                
                # Copy useful info from tomorrow's analysis
                for key in ["observable_start", "observable_end", "observable_hours", "best_time"]:
                    if key in tomorrow_visibility:
                        final_visibility[key] = tomorrow_visibility[key]
                
                # Mark as showing tomorrow
                final_visibility["showing_tomorrow"] = True
                final_visibility["tomorrow_date"] = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
                
                show_current_time = False  # No current time marker for tomorrow
                
            else:
                # Case 1 & 2: Use today's data
                plot_data = today_data
                final_visibility = today_visibility
                show_current_time = (status == "observable_now")  # Only show current time if observable now
            
            # Step 3: Create output file path
            if test_mode:
                test_dir = "./test_plots"
                os.makedirs(test_dir, exist_ok=True)
                filename = f"{grb_name.replace(' ', '_') if grb_name else 'target'}_visibility_{int(time.time())}.png"
                temp_path = os.path.join(test_dir, filename)
            else:
                temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
                os.close(temp_fd)
            
            # Step 4: Create the plot
            plt.figure(dpi=300, figsize=(10, 4))
            self.staralt.plot_staralt(data=plot_data, show_current_time=show_current_time)
            
            # Step 5: Add tomorrow warning label if needed
            if final_visibility.get("showing_tomorrow"):
                tomorrow_date = final_visibility.get("tomorrow_date", "Next Night")
                plt.figtext(0.5, 0.95, f"⚠️ SHOWING TOMORROW'S SKY ({tomorrow_date}) ⚠️", 
                           ha='center', va='center', fontsize=12, weight='bold',
                           bbox=dict(facecolor='yellow', alpha=0.7, boxstyle='round'))
            
            # Step 6: Save plot
            if savefig:
                plt.savefig(temp_path, bbox_inches='tight')
                plt.close()
            
            self.logger.info(f"Successfully created visibility plot for {grb_name or 'target'}")
            return temp_path, final_visibility
            
        except Exception as e:
            self.logger.error(f"Error creating visibility plot: {e}", exc_info=True)
            return None, {"status": "error", "message": str(e)}

    def format_visibility_message(self, visibility_info: Dict[str, Any]) -> str:
        """
        Format visibility information into a structured message for Slack.
        """
        try:
            # Extract basic info
            status = visibility_info.get("status", "unknown")
            condition = visibility_info.get("condition", "Unknown")
            
            # Start building message
            sections = []
            
            # Format header based on status with color emoji
            if status == "observable_now":
                header = "*🟢 CURRENTLY OBSERVABLE*"
                sections.append(f"{header}")
            elif status == "observable_later":
                header = "*🟠 OBSERVABLE LATER TONIGHT*"
                sections.append(f"{header}")
            elif status == "observable_tomorrow":
                header = "*🔵 OBSERVABLE TOMORROW NIGHT*"
                sections.append(f"{header}")
            else:
                header = "*🔴 NOT OBSERVABLE*"
                sections.append(f"{header}")
            
            # Add condition
            if condition != "Unknown":
                sections.append(f"> - 🌃 *Condition*: {condition}")
            
            # Add detailed information based on status
            if status == "observable_now":
                # Currently observable details
                end_time_obj = visibility_info.get("observable_end")
                end_time = self._format_time_clt_kst(end_time_obj) if end_time_obj else "Unknown"
                
                remaining = visibility_info.get("remaining_hours", 0)
                alt = visibility_info.get("current_altitude", 0)
                moon_sep = visibility_info.get("current_moon_separation", 0)
                
                details = [
                    f"> - ⏰ *Observable now until*: {end_time} (*{remaining:.1f} hours* remaining)",
                    f"> - 📈 *Current altitude*: {alt:.1f}° (minimum required: 30°)",
                    f"> - 🌙 *Moon separation*: {moon_sep:.1f}° (minimum required: 30°)"
                ]
                sections.extend(details)
                
            elif status == "observable_later":
                # Observable later details
                start_time_obj = visibility_info.get("observable_start")
                end_time_obj = visibility_info.get("observable_end")
                
                start_time = self._format_time_clt_kst(start_time_obj) if start_time_obj else "Unknown"
                end_time = self._format_time_clt_kst(end_time_obj) if end_time_obj else "Unknown"
                
                hours_until = visibility_info.get("hours_until_observable", 0)
                window = visibility_info.get("observable_hours", 0)
                
                details = [
                    f"> - ⏱️ *Observable in*: {hours_until:.1f} hours (starts at {start_time})",
                    f"> - ⏰ *Observable window*: {start_time} to {end_time} (*{window:.1f} hours*)"
                ]
                sections.extend(details)
                
            elif status == "observable_tomorrow":
                # Tomorrow observability details
                reason = visibility_info.get("reason", "Check tomorrow night")
                
                # If we have tomorrow's window info, show it
                start_time_obj = visibility_info.get("observable_start")
                end_time_obj = visibility_info.get("observable_end")
                window = visibility_info.get("observable_hours", 0)
                
                details = [f"> - 📆 *Reason*: {reason}"]
                
                if start_time_obj and end_time_obj:
                    start_time = self._format_time_clt_kst(start_time_obj)
                    end_time = self._format_time_clt_kst(end_time_obj)
                    details.extend([
                        f"> - 🕙 *Tomorrow's window*: {start_time} to {end_time} (*{window:.1f} hours*)",
                        f"> - ⏳ *Check tomorrow*: ~24 hours from now"
                    ])
                else:
                    details.append(f"> - ⏳ *Check tomorrow*: ~24 hours from now")
                
                sections.extend(details)
                
            else:
                # Not observable details
                reason = visibility_info.get("reason", "Unknown limitation")
                sections.append(f"> - ❌ *Reason*: {reason}")
            
            # Combine all sections
            return "\n".join(sections)
            
        except Exception as e:
            self.logger.error(f"Error formatting visibility message: {e}", exc_info=True)
            return f"*Visibility Analysis Error*\nCould not format visibility information: {str(e)}"