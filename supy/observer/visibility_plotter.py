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
    
    def _analyze_visibility(self, staralt_data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze visibility data for a target to determine if it's observable.
        Provides detailed information for GRB observation planning.
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
            "message": "Not observable tonight",
            "recommendation": "Observation not possible"
        }
        
        try:
            # Extract relevant information from staralt_data_dict
            now_datetime = staralt_data_dict.get("now_datetime")
            color_target = staralt_data_dict.get("color_target", [])
            target_times = staralt_data_dict.get("target_times", [])
            target_alts = staralt_data_dict.get("target_alts", [])
            target_moonsep = staralt_data_dict.get("target_moonsep", [])
            min_altitude = staralt_data_dict.get("target_minalt", 30)
            min_moon_sep = staralt_data_dict.get("target_minmoonsep", 30)
            
            # If data is missing, return early
            if not all([now_datetime, color_target, target_times, target_alts, target_moonsep]):
                result["reason"] = "Insufficient data for visibility analysis"
                return result
            
            # Convert target_times to datetime objects if they're strings
            target_times_dt = []
            for time_str in target_times:
                if isinstance(time_str, str):
                    target_times_dt.append(datetime.fromisoformat(time_str))
                else:
                    target_times_dt.append(time_str)
            
            # Find observable periods (sequences of 'g' in color_target)
            observable_indices = [i for i, color in enumerate(color_target) if color == 'g']
            
            if not observable_indices:
                # Target is not observable tonight
                # Determine reason why it's not observable
                
                # Get maximum altitude during the night
                max_alt = max(target_alts) if target_alts else 0
                min_moonsep = min(target_moonsep) if target_moonsep else 0
                
                # Determine if the target might be observable tomorrow
                might_be_observable_tomorrow = False
                
                # Check if the target almost meets altitude requirements
                # If it's close to the minimum altitude but not quite there
                if 0 < max_alt < min_altitude and max_alt > min_altitude - 15:
                    might_be_observable_tomorrow = True
                    
                # Check if moon separation is the issue and moon will move
                if max_alt >= min_altitude and min_moonsep < min_moon_sep and min_moonsep > min_moon_sep - 20:
                    might_be_observable_tomorrow = True
                
                if might_be_observable_tomorrow:
                    result["status"] = "observable_tomorrow"
                    result["condition"] = "Likely Observable Tomorrow"
                    
                    if max_alt < min_altitude:
                        result["reason"] = f"Target reaches {max_alt:.1f}° (close to minimum {min_altitude}°), may be higher tomorrow"
                    elif min_moonsep < min_moon_sep:
                        result["reason"] = f"Target too close to Moon (min separation: {min_moonsep:.1f}°), moon position will change tomorrow"
                    
                    result["recommendation"] = "Check visibility for tomorrow night"
                else:
                    # Regular not observable status
                    if max_alt < min_altitude:
                        if max_alt <= 0:
                            result["condition"] = "Never Rises"
                            result["reason"] = f"Target never rises above horizon from Chile"
                        else:
                            result["condition"] = "Below Minimum Altitude"
                            result["reason"] = f"Target maximum altitude ({max_alt:.1f}°) below minimum required ({min_altitude}°)"
                    elif min_moonsep < min_moon_sep:
                        result["condition"] = "Moon Interference"
                        result["reason"] = f"Target too close to Moon (min separation: {min_moonsep:.1f}°, required: {min_moon_sep}°)"
                    
                    result["recommendation"] = "Observation not possible from Chile"
                
                return result
            
            # Get start and end of observable period
            start_idx = observable_indices[0]
            end_idx = observable_indices[-1]
            start_time = target_times_dt[start_idx]
            end_time = target_times_dt[end_idx]
            
            # Find time of maximum altitude during observable period
            max_alt_idx = start_idx + target_alts[start_idx:end_idx+1].index(max(target_alts[start_idx:end_idx+1]))
            best_time = target_times_dt[max_alt_idx]
            
            # Calculate total observable hours
            total_hours = (end_time - start_time).total_seconds() / 3600
            
            # Store these in results
            result["observable_hours"] = round(total_hours, 1)
            result["observable_start"] = start_time
            result["observable_end"] = end_time
            result["best_time"] = best_time
            
            # Find the closest time index to now
            now_idx = min(range(len(target_times_dt)), 
                          key=lambda i: abs((target_times_dt[i] - now_datetime).total_seconds()))
            
            # Get current altitude and moon separation
            result["current_altitude"] = target_alts[now_idx]
            result["current_moon_separation"] = target_moonsep[now_idx]
            
            # Determine if target is currently observable
            if now_idx in observable_indices:
                # Currently observable
                result["status"] = "observable_now"
                
                # Determine condition based on altitude and remaining time
                if result["current_altitude"] > 60:
                    result["condition"] = "Excellent Observing Conditions"
                elif result["current_altitude"] > 45:
                    result["condition"] = "Good Observing Conditions"
                else:
                    result["condition"] = "Acceptable Observing Conditions"
                
                # Calculate remaining observable time
                remaining_hours = (end_time - now_datetime).total_seconds() / 3600
                result["remaining_hours"] = round(remaining_hours, 1)
                
                if remaining_hours < 1:
                    result["condition"] = "Limited Time Remaining"
                
                # Create message and recommendation
                result["message"] = (
                    f"Target currently at {result['current_altitude']:.1f}° altitude with "
                    f"{result['current_moon_separation']:.1f}° Moon separation.\n"
                    f"Observable for {result['remaining_hours']} more hours (until "
                    f"{self._format_time_clt_kst(result['observable_end'])})."
                )
                
                result["recommendation"] = "Begin observations immediately"
                
            else:
                # Check if target will be observable later tonight
                if now_datetime < start_time:
                    result["status"] = "observable_later"
                    
                    # Calculate hours until observable
                    hours_until = (start_time - now_datetime).total_seconds() / 3600
                    result["hours_until_observable"] = round(hours_until, 1)
                    
                    # Determine condition based on wait time
                    if hours_until < 1:
                        result["condition"] = "Observable Very Soon"
                    elif hours_until < 3:
                        result["condition"] = "Observable in a Few Hours"
                    else:
                        result["condition"] = "Long Wait for Observation"
                    
                    # Create message and recommendation
                    result["message"] = (
                        f"Target will be observable in {result['hours_until_observable']} hours (starting at "
                        f"{self._format_time_clt_kst(result['observable_start'])}).\n"
                        f"Observable window: {self._format_time_clt_kst(result['observable_start'])} to "
                        f"{self._format_time_clt_kst(result['observable_end'])} ({result['observable_hours']} hours).\n"
                        f"Best observation time: {self._format_time_clt_kst(result['best_time'])}"
                    )
                    
                    result["recommendation"] = f"Schedule observations to begin at {self._format_time_clt_kst(result['observable_start'])}"
                else:
                    # Target was observable earlier but not anymore

                    if now_datetime > end_time:
                        result["status"] = "not_observable"
                        result["condition"] = "Observation Window Passed"
                        result["reason"] = f"The astronomcal dawn has passed"
                        result["recommendation"] = "Observation no longer possible tonight"

                    else:
                        # if result["current_altitude"] < min_altitude:
                            result["status"] = "not_observable"
                            result["condition"] = "Observation Window Passed"
                            result["reason"] = f"Target was observable earlier but has now set below the {min_altitude}° altitude limit"
                            result["recommendation"] = "Observation no longer possible tonight"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing visibility data: {e}", exc_info=True)
            result["reason"] = f"Error in visibility analysis: {str(e)}"
            return result
    
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
                # sections.append(f"{header}\n➡ *Immediate Observation POSSIBLE*")
            elif status == "observable_later":
                header = "*🟠 OBSERVABLE LATER TONIGHT*"
                sections.append(f"{header}")
                # sections.append(f"{header}\n➡ *DELAYED Observation Required*")
            elif status == "observable_tomorrow":
                header = "*🔵 LIKELY OBSERVABLE TOMORROW*"
                sections.append(f"{header}")
                # sections.append(f"{header}\n➡ *TOMORROW Observation Recommended*")
            else:
                header = "*🔴 NOT OBSERVABLE*"
                sections.append(f"{header}")
                # sections.append(f"{header}\n➡ *Observation NOT POSSIBLE from CHILE*")
            
            # Add condition
            if condition != "Unknown":
                sections.append(f"> - 🌃 *Condition*: {condition}")
            
            # Add detailed information based on status
            if status == "observable_now":
                # Currently observable details
                # Safely get observable_end with fallback
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
                # Safely get time objects with fallbacks
                start_time_obj = visibility_info.get("observable_start")
                end_time_obj = visibility_info.get("observable_end")
                best_time_obj = visibility_info.get("best_time")
                
                # Format times with fallbacks
                start_time = self._format_time_clt_kst(start_time_obj) if start_time_obj else "Unknown"
                end_time = self._format_time_clt_kst(end_time_obj) if end_time_obj else "Unknown"
                best_time = self._format_time_clt_kst(best_time_obj) if best_time_obj else "Unknown"
                
                hours_until = visibility_info.get("hours_until_observable", 0)
                window = visibility_info.get("observable_hours", 0)
                
                details = [
                    f"> - ⏱️ *Observable in*: {hours_until:.1f} hours (starts at {start_time})",
                    f"> - ⏰ *Observable window*: {start_time} to {end_time} (*{window:.1f} hours*)",
                    f"> - ⭐ *Best observation time*: {best_time} (highest altitude)"
                ]
                sections.extend(details)
            elif status == "observable_tomorrow":
                # Enhanced tomorrow observability details
                reason = visibility_info.get("reason", "Target may be observable tomorrow")
                
                # Calculate tomorrow's timing information using existing data
                # Get the dawn/dusk times from today and add 24 hours
                now = datetime.now()
                
                # Get approximate observation window tomorrow
                tomorrow_start_time = now + timedelta(hours=24)
                tomorrow_start_formatted = self._format_time_clt_kst(tomorrow_start_time)
                
                # Estimate window duration based on current max altitude (higher = longer window)
                max_alt = visibility_info.get("max_altitude", 45)  # Default to 45 degrees
                est_duration = 0
                
                if max_alt < 30:
                    est_duration = 1.5  # Very short window
                elif max_alt < 45:
                    est_duration = 3.0  # Medium window
                else:
                    est_duration = 4.5  # Long window
                
                # Calculate approximate end time
                tomorrow_end_time = tomorrow_start_time + timedelta(hours=est_duration)
                tomorrow_end_formatted = self._format_time_clt_kst(tomorrow_end_time)
                
                # Calculate hours until tomorrow's observation
                hours_until_tomorrow = 24 + visibility_info.get("hours_until_observable", 0)
                if hours_until_tomorrow > 24:
                    hours_until_tomorrow = 24  # Cap at 24 hours if greater
                
                details = [
                    f"> - 📆 *Future observability*: {reason}",
                    f"> - 🕙 *Estimated tomorrow window*: {tomorrow_start_formatted} to {tomorrow_end_formatted} (*~{est_duration:.1f} hours*)",
                    f"> - ⏳ *Hours until observable*: ~{hours_until_tomorrow:.1f} hours from now"
                ]
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

    def create_visibility_plot(self, ra, dec, grb_name=None, test_mode=False, minalt=30, minmoonsep=30, savefig=True):
        """
        Create a visibility plot for given coordinates.
        
        Args:
            ra: Right Ascension in degrees
            dec: Declination in degrees
            grb_name: Name of the GRB for plot title
            test_mode: If True, save plot to test_plots directory instead of temp file
            minalt: Minimum altitude for target in degrees
            minmoonsep: Minimum moon separation for target in degrees
            
        Returns:
            tuple: (path_to_plot_file, visibility_info) or (None, visibility_info) if not observable
        """
        try:
            # First get initial visibility data to determine status
            self.staralt.set_target(
                ra=ra,
                dec=dec,
                objname=grb_name,
                target_minalt=minalt,
                target_minmoonsep=minmoonsep
            )
            
            # Analyze visibility
            visibility_info = self._analyze_visibility(self.staralt.data_dict)
            
            # If visibility status is "observable_tomorrow", create a plot for tomorrow
            if visibility_info.get("status") == "observable_tomorrow":
                self.logger.info(f"Target likely observable tomorrow - generating tomorrow's sky plot")
                
                # Use a date that's 24 hours in the future
                tomorrow = datetime.now() + timedelta(days=1)
                
                # Generate visibility data for tomorrow
                self.staralt.set_target(
                    ra=ra,
                    dec=dec,
                    objname=grb_name if grb_name else "Target",
                    utctime=tomorrow,  # Use tomorrow as the reference time
                    target_minalt=minalt,
                    target_minmoonsep=minmoonsep
                )
                
                # Re-analyze visibility with tomorrow's data
                tomorrow_visibility_info = self._analyze_visibility(self.staralt.data_dict)
                
                # Add showing_tomorrow flag to tomorrow's data
                visibility_info["showing_tomorrow"] = True
                visibility_info["tomorrow_date"] = tomorrow.strftime("%Y-%m-%d")
                
                # Copy useful visibility data from tomorrow's analysis
                # but keep the original "observable_tomorrow" status
                for key in ["observable_start", "observable_end", "observable_hours", "best_time"]:
                    if key in tomorrow_visibility_info:
                        visibility_info[key] = tomorrow_visibility_info[key]
            
            # If not observable, return early with no plot
            if visibility_info.get("status") == "not_observable":
                self.logger.info(f"Target not observable. Status: {visibility_info.get('status')}. No plot generated.")
                return None, visibility_info
                
            # Create output file path
            if test_mode:
                # Use test directory
                test_dir = "./test_plots"
                os.makedirs(test_dir, exist_ok=True)
                
                if grb_name:
                    filename = f"{grb_name.replace(' ', '_')}_visibility_{int(time.time())}.png"
                else:
                    filename = f"visibility_plot_{int(time.time())}.png"
                
                temp_path = os.path.join(test_dir, filename)
            else:
                # Use tempfile
                temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
                os.close(temp_fd)
            
            # Determine if we should show the current time marker
            # Only show for "observable_now" status, not for "observable_later" or "observable_tomorrow"
            show_current_time = visibility_info.get("status") == "observable_now"
            
            # Create plot
            plt.figure(dpi=300, figsize=(10, 4))
            self.staralt.plot_staralt(show_current_time=show_current_time)
            
            # Add a label if we're showing tomorrow's sky
            if visibility_info.get("showing_tomorrow"):
                tomorrow_date = visibility_info.get("tomorrow_date", 
                            (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"))
                
                plt.figtext(0.5, 0.95, f"⚠️ SHOWING TOMORROW'S SKY ({tomorrow_date}) ⚠️", 
                        ha='center', va='center', fontsize=12, weight='bold',
                        bbox=dict(facecolor='yellow', alpha=0.5, boxstyle='round'))
            
            # Save plot
            if savefig:
                plt.savefig(temp_path, bbox_inches='tight')
                plt.close()
            
            self.logger.info(f"Successfully created visibility plot for {grb_name or 'target'}")
            return temp_path, visibility_info
            
        except Exception as e:
            self.logger.error(f"Error creating visibility plot: {e}", exc_info=True)
            return None, {"status": "error", "message": str(e)}

#---------------------------------------Test Code----------------------------------------
if __name__ == "__main__":
    # Configure logging only for standalone testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('./test/visibility_test.log'),
            logging.StreamHandler()
        ]
    )
    test_logger = logging.getLogger(__name__)
    test_logger.info("Starting enhanced visibility plotter test")
    
    # Create test directory if it doesn't exist
    test_dir = "test_plots"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # Initialize plotter
    plotter = VisibilityPlotter(logger=test_logger)
    
    # Test visibility time conversion functions
    test_logger.info("Testing timezone conversion functions")
    now_utc = datetime.now(pytz.utc)
    chile_time, korea_time = plotter._convert_time_to_clt_kst(now_utc)
    test_logger.info(f"Current time: UTC: {now_utc}, Chile: {chile_time}, Korea: {korea_time}")
    
    # Test the formatting function
    formatted_time = plotter._format_time_clt_kst(now_utc)
    test_logger.info(f"Formatted time: {formatted_time}")
    
    # Test coordinates scenarios:
    # 1. Currently observable
    # 2. Observable later tonight
    # 3. Not observable (northern hemisphere)
    test_scenarios = [
        {"ra": 180.0, "dec": -30.0, "name": "TEST_CURRENT"},    # Should be observable from Chile now
        {"ra": 90.0, "dec": -20.0, "name": "TEST_LATER"},       # Should be observable later in Chile
        {"ra": 200.0, "dec": 60.0, "name": "TEST_NORTHERN"}     # Northern hemisphere target
    ]
    
    # Test each scenario
    for scenario in test_scenarios:
        test_logger.info(f"Testing visibility scenario: {scenario['name']}")
        
        # Generate visibility plot
        plot_path, visibility_info = plotter.create_visibility_plot(
            ra=scenario["ra"],
            dec=scenario["dec"],
            grb_name=scenario["name"],
            test_mode=True
        )
        
        # Log visibility info
        test_logger.info(f"Visibility status: {visibility_info.get('status')}")
        test_logger.info(f"Visibility condition: {visibility_info.get('condition')}")
        
        # Test message formatting
        formatted_message = plotter.format_visibility_message(visibility_info)
        test_logger.info(f"Formatted message:\n{formatted_message}")
        
        if plot_path:
            test_logger.info(f"Plot created at: {plot_path}")
        else:
            test_logger.warning(f"No plot created for {scenario['name']}")
    
    test_logger.info("Enhanced visibility plotter test completed")