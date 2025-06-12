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

def setup_visibility_logger(log_filename='gcn_bot.log'):
    """
    Configure a logger for the visibility plotter that shares the same log file as gcn_bot.
    
    Args:
        log_filename (str): Name of the log file to write to
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger('visibility_plotter')
    
    # Clear existing handlers to avoid duplication
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatter with detailed information
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s'
    )
    
    # File handler - same file as gcn_bot
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    
    # Stream handler for console output
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    # Set level
    logger.setLevel(logging.INFO)
    
    # Prevent propagation to avoid duplicate messages
    logger.propagate = False
    
    return logger

# Initialize standalone logger for testing
standalone_logger = setup_visibility_logger()

class VisibilityPlotter:
    """
    Handle visibility plot generation and Slack uploading for GCN notices.
    
    This class creates visibility plots using staralt.py and handles 
    temporary file management for uploading to Slack. It provides specialized
    visibility analysis for GRB observations from Chile.
    """
    
    def __init__(self, logger=None, log_filename='gcn_bot.log'):
        """
        Initialize the observer and plotter.
        
        Args:
            logger: Logger instance from main application (gcn_bot.py)
            log_filename: Name of the log file to use for standalone operation
        """
        self.observer = mainObserver()  # Use default parameters
        self.staralt = Staralt(self.observer)
        
        # Use provided logger or create our own that shares the same log file
        if logger is not None:
            self.logger = logger
            self.logger.info("VisibilityPlotter initialized with external logger")
        else:
            self.logger = setup_visibility_logger(log_filename)
            self.logger.info("VisibilityPlotter initialized with standalone logger")
        
        # Define timezones
        self.chile_tz = pytz.timezone("America/Santiago")
        self.korea_tz = pytz.timezone("Asia/Seoul")
        
        # Log initialization details
        self.logger.debug(f"Observer location: Lat {self.observer._latitude}, Lon {self.observer._longitude}")
        self.logger.debug(f"Timezone setup: Chile={self.chile_tz}, Korea={self.korea_tz}")
    
    def _convert_time_to_clt_kst(self, utc_time: datetime) -> Tuple[datetime, datetime]:
        """
        Convert UTC time to Chile local time and Korean time.
        
        Args:
            utc_time: Datetime in UTC
            
        Returns:
            Tuple containing (chile_time, korea_time)
        """
        try:
            # Ensure UTC time has timezone info
            if utc_time.tzinfo is None:
                utc_time = pytz.utc.localize(utc_time)
                
            # Convert to Chile and Korea times
            chile_time = utc_time.astimezone(self.chile_tz)
            korea_time = utc_time.astimezone(self.korea_tz)
            
            self.logger.debug(f"Time conversion - UTC: {utc_time}, CLT: {chile_time}, KST: {korea_time}")
            return chile_time, korea_time
            
        except Exception as e:
            self.logger.error(f"Error converting time zones: {e}")
            raise
    
    def _format_time_clt_kst(self, utc_time: Optional[datetime]) -> str:
        """
        Format time in both CLT and KST timezones.
        
        Args:
            utc_time: Datetime in UTC, or None
            
        Returns:
            String with formatted time in both timezones, or "Unknown" if utc_time is None
        """
        if utc_time is None:
            self.logger.debug("Time formatting requested for None datetime")
            return "Unknown"
            
        try:
            chile_time, korea_time = self._convert_time_to_clt_kst(utc_time)
            formatted_time = f"{chile_time.strftime('%H:%M')} CLT / {korea_time.strftime('%H:%M')} KST"
            self.logger.debug(f"Formatted time: {formatted_time}")
            return formatted_time
        except Exception as e:
            self.logger.error(f"Error formatting time: {e}")
            return "Error formatting time"
    
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
            self.logger.info("Starting visibility status analysis")
            
            # Extract data from staralt analysis
            now_datetime = staralt_data_dict.get("now_datetime")
            color_target = staralt_data_dict.get("color_target", [])
            target_times = staralt_data_dict.get("target_times", [])
            target_alts = staralt_data_dict.get("target_alts", [])
            target_moonsep = staralt_data_dict.get("target_moonsep", [])
            min_altitude = staralt_data_dict.get("target_minalt", 30)
            min_moon_sep = staralt_data_dict.get("target_minmoonsep", 30)
            tonight = staralt_data_dict.get("tonight", {})
            
            # Extract sunset/sunrise for debugging
            sunset_night = tonight.get("sunset_night")
            sunrise_night = tonight.get("sunrise_night")
            
            # ENHANCED DEBUGGING
            self.logger.debug(f"=== VISIBILITY ANALYSIS DEBUG ===")
            self.logger.debug(f"Current time: {now_datetime}")
            self.logger.debug(f"Current date: {now_datetime.date() if now_datetime else 'None'}")
            self.logger.debug(f"Tonight's sunset: {sunset_night}")
            self.logger.debug(f"Tonight's sunrise: {sunrise_night}")
            
            if sunset_night and now_datetime:
                # Handle both datetime objects and astropy Time objects
                if hasattr(sunset_night, 'datetime'):
                    sunset_date = sunset_night.datetime.date()
                elif hasattr(sunset_night, 'date'):
                    sunset_date = sunset_night.date()
                else:
                    sunset_date = sunset_night
                    
                current_date = now_datetime.date()
                date_diff = (current_date - sunset_date).days
                self.logger.debug(f"Date difference (current - sunset): {date_diff} days")
                
                if date_diff > 1:
                    self.logger.error(f"CRITICAL: Using outdated night window! Current: {current_date}, Sunset: {sunset_date}")
                elif date_diff < 0:
                    self.logger.warning(f"WARNING: Sunset date is in the future! Current: {current_date}, Sunset: {sunset_date}")
            
            # Additional validation
            if sunset_night and sunrise_night:
                # Check if we're using a reasonable night window
                if hasattr(sunset_night, 'datetime') and hasattr(sunrise_night, 'datetime'):
                    night_duration = (sunrise_night.datetime - sunset_night.datetime).total_seconds() / 3600
                    self.logger.debug(f"Night duration: {night_duration:.1f} hours")
                    
                    if night_duration < 8 or night_duration > 16:
                        self.logger.warning(f"Unusual night duration: {night_duration:.1f} hours")
            
            # Validate required data
            if not target_times or not target_alts or not target_moonsep:
                result["reason"] = "Missing target data"
                self.logger.error("Missing required target data for analysis")
                return result
            
            if not now_datetime:
                result["reason"] = "Missing current time"
                self.logger.error("Current time not available for analysis")
                return result
            
            # Convert target times to datetime objects for comparison
            target_times_dt = []
            for time_str in target_times:
                try:
                    if isinstance(time_str, str):
                        dt = datetime.fromisoformat(time_str)
                    else:
                        dt = time_str
                    target_times_dt.append(dt)
                except (ValueError, TypeError) as e:
                    self.logger.error(f"Error converting time {time_str}: {e}")
                    continue
            
            if not target_times_dt:
                result["reason"] = "No valid target times"
                self.logger.error("No valid target times for analysis")
                return result
            
            # Find current time index
            now_idx = None
            for i, dt in enumerate(target_times_dt):
                if dt >= now_datetime:
                    now_idx = i
                    break
            
            if now_idx is None:
                now_idx = len(target_times_dt) - 1
            
            # Find observable periods (both altitude and moon separation criteria)
            observable_indices = []
            for i, (alt, moon_sep) in enumerate(zip(target_alts, target_moonsep)):
                if alt >= min_altitude and moon_sep >= min_moon_sep:
                    observable_indices.append(i)
            
            # Log current conditions
            if now_idx < len(target_alts) and now_idx < len(target_moonsep):
                current_alt = target_alts[now_idx]
                current_moon = target_moonsep[now_idx]
                result["current_altitude"] = current_alt
                result["current_moon_separation"] = current_moon
                self.logger.info(f"Current conditions - Alt: {current_alt:.1f}°, Moon sep: {current_moon:.1f}°")
            
            self.logger.info(f"Found {len(observable_indices)} observable time points")
            self.logger.debug(f"Observable indices: {observable_indices}")
            self.logger.debug(f"Current time index: {now_idx}")
            self.logger.debug(f"=== END DEBUG ===")
            
            # Case 1: Currently observable
            if observable_indices and now_idx in observable_indices:
                self.logger.info("Case 1: Currently observable")
                return self._handle_observable_now(result, observable_indices, target_times_dt, now_datetime)
            
            # Case 2: Observable later tonight
            elif observable_indices and any(i > now_idx for i in observable_indices):
                self.logger.info("Case 2: Observable later tonight")
                return self._handle_observable_later(result, observable_indices, target_times_dt, now_datetime, now_idx)
                
            # Case 3 & 4: Not observable tonight - check if tomorrow is possible
            else:
                self.logger.info("Case 3/4: Not observable tonight - checking tomorrow possibility")
                return self._handle_not_observable_tonight(result, target_alts, target_moonsep, min_altitude, min_moon_sep, tonight, now_datetime)
                    
        except Exception as e:
            self.logger.error(f"Error analyzing visibility: {e}", exc_info=True)
            result["reason"] = f"Error in visibility analysis: {str(e)}"
            return result
    
    def _handle_observable_now(self, result: Dict[str, Any], observable_indices: List[int], 
                              target_times_dt: List[datetime], now_datetime: datetime) -> Dict[str, Any]:
        """Handle Case 1: Currently observable"""
        self.logger.info("Processing currently observable target")
        
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
        
        self.logger.info(f"Observable window: {result['observable_hours']:.1f}h total, "
                        f"{result['remaining_hours']:.1f}h remaining")
        
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
        
        self.logger.info(f"Observable now - Condition: {result['condition']}")
        return result
    
    def _handle_observable_later(self, result: Dict[str, Any], observable_indices: List[int], 
                                target_times_dt: List[datetime], now_datetime: datetime, now_idx: int) -> Dict[str, Any]:
        """Handle Case 2: Observable later tonight"""
        self.logger.info("Processing target observable later tonight")
        
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
        
        self.logger.info(f"Observable in {hours_until:.1f}h for {result['observable_hours']:.1f}h")
        
        # Set condition based on wait time
        if hours_until < 1:
            result["condition"] = "Observable Very Soon"
        elif hours_until < 3:
            result["condition"] = "Observable in a Few Hours"
        else:
            result["condition"] = "Long Wait for Observation"
        
        result["recommendation"] = f"Schedule observations to begin at {self._format_time_clt_kst(result['observable_start'])}"
        
        self.logger.info(f"Observable later - Condition: {result['condition']}")
        return result
    
    def _handle_not_observable_tonight(self, result: Dict[str, Any], target_alts: List[float], 
                                      target_moonsep: List[float], min_altitude: float, min_moon_sep: float,
                                      tonight: Dict[str, Any], now_datetime: datetime) -> Dict[str, Any]:
        """Handle Cases 3 & 4: Not observable tonight"""
        self.logger.info("Processing target not observable tonight")
        
        # Get tonight's observing window times
        sunset_night = tonight.get("sunset_night")
        sunrise_night = tonight.get("sunrise_night")
        
        # Check if we're past tonight's observing window
        is_past_observing_window = False
        if sunset_night and sunrise_night and now_datetime:
            # If current time is past sunrise (astronomical dawn), tonight's window is over
            if now_datetime > sunrise_night:
                is_past_observing_window = True
                self.logger.info("Current time is past tonight's observing window")
        
        # Get maximum conditions tonight
        max_alt = max(target_alts) if target_alts else 0
        min_moonsep_tonight = min(target_moonsep) if target_moonsep else 0
        
        self.logger.info(f"Tonight's max conditions - Alt: {max_alt:.1f}°, Moon sep: {min_moonsep_tonight:.1f}°")
        
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
            self.logger.info(f"Observable tomorrow - Reason: {tomorrow_reason}")
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
        self.logger.info(f"Not observable - {result['condition']}: {result['reason']}")
        
        return result

    def _generate_today_visibility(self, ra, dec, grb_name, minalt, minmoonsep) -> Tuple[Dict[str, Any], Any]:
        """Generate visibility data for today"""
        self.logger.info(f"Generating today's visibility for {grb_name} at RA={ra:.2f}, DEC={dec:.2f}")
        
        try:
            # Use current time for today's analysis
            current_time = datetime.now()
            self.logger.debug(f"Using current time for today's analysis: {current_time}")
            
            self.staralt.set_target(
                ra=ra,
                dec=dec,
                objname=grb_name,
                utctime=current_time,  # Ensure we use current time
                target_minalt=minalt,
                target_minmoonsep=minmoonsep
            )
            
            # VALIDATION: Check that calculated night times are reasonable
            data_dict = self.staralt.data_dict
            tonight = data_dict.get("tonight", {})
            sunset_night = tonight.get("sunset_night")
            sunrise_night = tonight.get("sunrise_night")
            
            if sunset_night and sunrise_night:
                current_date = current_time.date()
                sunset_date = sunset_night.date() if hasattr(sunset_night, 'date') else sunset_night
                
                # Sunset should be today or yesterday (for early morning observations)
                date_diff = (current_date - sunset_date).days
                if date_diff > 1:
                    self.logger.error(f"Invalid night calculation: sunset {sunset_date} too old for current date {current_date}")
                    raise ValueError(f"Night time calculation error: sunset date mismatch")
            
            today_visibility = self._analyze_visibility_status(self.staralt.data_dict)
            self.logger.info(f"Today's visibility analysis complete - Status: {today_visibility.get('status')}")
            return today_visibility, self.staralt.data
            
        except Exception as e:
            self.logger.error(f"Error generating today's visibility: {e}")
            raise

    def _generate_tomorrow_visibility(self, ra: float, dec: float, grb_name: str, minalt: float, minmoonsep: float) -> Tuple[Dict[str, Any], Any]:
        """Generate visibility data for tomorrow night"""
        self.logger.info(f"Generating tomorrow's visibility for {grb_name}")
        
        try:
            # Calculate tomorrow's date (next observing night)
            tomorrow = datetime.now() + timedelta(days=1)
            self.logger.debug(f"Tomorrow's date: {tomorrow.strftime('%Y-%m-%d')}")
            
            self.staralt.set_target(
                ra=ra,
                dec=dec,
                objname=grb_name,
                utctime=tomorrow,
                target_minalt=minalt,
                target_minmoonsep=minmoonsep
            )
            
            tomorrow_visibility = self._analyze_visibility_status(self.staralt.data_dict)
            self.logger.info(f"Tomorrow's visibility analysis complete - Status: {tomorrow_visibility.get('status')}")
            return tomorrow_visibility, self.staralt.data
            
        except Exception as e:
            self.logger.error(f"Error generating tomorrow's visibility: {e}")
            raise

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
        target_name = grb_name or "Target"
        self.logger.info(f"Creating visibility plot for {target_name} at RA={ra:.3f}, DEC={dec:.3f}")
        self.logger.info(f"Plot parameters - MinAlt: {minalt}°, MinMoonSep: {minmoonsep}°, TestMode: {test_mode}")
        
        try:
            # Step 1: Analyze today's visibility
            self.logger.info("Step 1: Analyzing today's visibility")
            today_visibility, today_data = self._generate_today_visibility(ra, dec, target_name, minalt, minmoonsep)
            
            status = today_visibility.get("status")
            self.logger.info(f"Visibility status determined: {status}")
            
            # Step 2: Handle each case
            if status == "not_observable":
                # Case 4: No plot
                self.logger.info("Case 4: Target not observable - no plot will be generated")
                return None, today_visibility
            
            elif status == "observable_tomorrow":
                # Case 3: Generate tomorrow's plot
                self.logger.info("Case 3: Generating tomorrow's visibility plot")
                tomorrow_visibility, tomorrow_data = self._generate_tomorrow_visibility(ra, dec, target_name, minalt, minmoonsep)
                
                # The staralt object now has tomorrow's data loaded
                final_visibility = today_visibility.copy()
                
                # Copy useful info from tomorrow's analysis
                for key in ["observable_start", "observable_end", "observable_hours", "best_time"]:
                    if key in tomorrow_visibility:
                        final_visibility[key] = tomorrow_visibility[key]
                
                # Mark as showing tomorrow
                final_visibility["showing_tomorrow"] = True
                final_visibility["tomorrow_date"] = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
                
                show_current_time = False  # No current time marker for tomorrow
                self.logger.info("Tomorrow's plot will be generated without current time marker")
                
            else:
                # Case 1 & 2: Use today's data (already loaded in staralt object)
                final_visibility = today_visibility
                show_current_time = (status == "observable_now")  # Only show current time if observable now
                self.logger.info(f"Today's plot will be generated - Show current time: {show_current_time}")
            
            # Step 3: Create output file path
            if test_mode:
                test_dir = "./test_plots"
                os.makedirs(test_dir, exist_ok=True)
                filename = f"{target_name.replace(' ', '_')}_visibility_{int(time.time())}.png"
                temp_path = os.path.join(test_dir, filename)
                self.logger.info(f"Test mode: Plot will be saved to {temp_path}")
            else:
                temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
                os.close(temp_fd)
                self.logger.debug(f"Production mode: Plot will be saved to temporary file {temp_path}")
            
            # Step 4: Create the plot (staralt object has the correct data loaded)
            self.logger.info("Step 4: Generating plot with staralt")
            plt.figure(dpi=300, figsize=(10, 4))
            self.staralt.plot_staralt(show_current_time=show_current_time)
            
            # Step 5: Add tomorrow warning label if needed
            if final_visibility.get("showing_tomorrow"):
                tomorrow_date = final_visibility.get("tomorrow_date", "Next Night")
                self.logger.info(f"Adding tomorrow warning label for date: {tomorrow_date}")
                plt.figtext(0.5, 0.95, f"⚠️ SHOWING TOMORROW'S SKY ({tomorrow_date}) ⚠️", 
                           ha='center', va='center', fontsize=12, weight='bold',
                           bbox=dict(facecolor='yellow', alpha=0.7, boxstyle='round'))
            
            # Step 6: Save plot
            if savefig:
                self.logger.info("Step 6: Saving plot to file")
                plt.savefig(temp_path, bbox_inches='tight')
                plt.close()
                self.logger.info(f"Plot successfully saved to {temp_path}")
            else:
                self.logger.info("Step 6: Skipping plot save (savefig=False)")
            
            self.logger.info(f"Visibility plot creation completed successfully for {target_name}")
            return temp_path, final_visibility
            
        except Exception as e:
            self.logger.error(f"Error creating visibility plot for {target_name}: {e}", exc_info=True)
            return None, {"status": "error", "message": str(e)}

    def format_visibility_message(self, visibility_info: Dict[str, Any]) -> str:
        """
        Format visibility information into a structured message for Slack.
        """
        try:
            self.logger.debug("Formatting visibility message for Slack")
            
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
                self.logger.debug(f"Formatted observable_now details: {remaining:.1f}h remaining")
                
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
                self.logger.debug(f"Formatted observable_later details: in {hours_until:.1f}h for {window:.1f}h")
                
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
                self.logger.debug(f"Formatted observable_tomorrow details: {reason}")
                
            else:
                # Not observable details
                reason = visibility_info.get("reason", "Unknown limitation")
                sections.append(f"> - ❌ *Reason*: {reason}")
                self.logger.debug(f"Formatted not_observable details: {reason}")
            
            # Combine all sections
            formatted_message = "\n".join(sections)
            self.logger.debug(f"Visibility message formatted successfully ({len(sections)} sections)")
            return formatted_message
            
        except Exception as e:
            self.logger.error(f"Error formatting visibility message: {e}", exc_info=True)
            return f"*Visibility Analysis Error*\nCould not format visibility information: {str(e)}"


# For backward compatibility and testing
if __name__ == "__main__":
    # Configure logging for standalone testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('./test_plots/visibility_test.log'),
            logging.StreamHandler()
        ]
    )
    test_logger = logging.getLogger(__name__)

    test_logger.info("\nTesting visibility plotter...")
    
    # Initialize plotter with test logger
    plotter = VisibilityPlotter(logger=test_logger)
    
    # Test coordinates (example GRB position)
    test_ra = 94.224
    test_dec = 56.893
    test_name = "Test GRB 250322A"
    
    test_logger.info(f"Testing with coordinates: RA={test_ra}, DEC={test_dec}")
    
    # Test visibility plot creation
    try:
        plot_path, visibility_info = plotter.create_visibility_plot(
            ra=test_ra,
            dec=test_dec,
            grb_name=test_name,
            test_mode=True,
            minalt=30,
            minmoonsep=30
        )
        
        if plot_path:
            test_logger.info(f"Plot created successfully: {plot_path}")
        else:
            test_logger.info("No plot created (target not observable)")
        
        # Test message formatting
        message = plotter.format_visibility_message(visibility_info)
        test_logger.info(f"Formatted message:\n{message}")
        
    except Exception as e:
        test_logger.error(f"Test failed: {e}", exc_info=True)
    
    test_logger.info("Visibility plotter test completed")