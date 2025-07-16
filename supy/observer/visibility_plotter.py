import os
import tempfile
import time
from datetime import datetime, timedelta
from .mainobserver import mainObserver
from .staralt import Staralt
import matplotlib.pyplot as plt
import logging
import pytz
from astropy.time import Time
from astropy import units as u
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
        # Use provided logger or create our own that shares the same log file
        if logger is not None:
            self.logger = logger
            self.logger.info("VisibilityPlotter initialized with external logger")
        else:
            self.logger = setup_visibility_logger(log_filename)
            self.logger.info("VisibilityPlotter initialized with standalone logger")

        # Initialize observer and staralt
        self.observer = mainObserver()  # Use default parameters
        self.staralt = Staralt(self.observer, logger=self.logger)

        # Define timezones
        self.chile_tz = pytz.timezone("America/Santiago")
        self.korea_tz = pytz.timezone("Asia/Seoul")
        
        # Log initialization details
        self.logger.debug(f"Observer location: Lat {self.observer._latitude}, Lon {self.observer._longitude}")
        self.logger.debug(f"Timezone setup: Chile={self.chile_tz}, Korea={self.korea_tz}")
    
    def _convert_time_to_clt_kst(self, utc_time: Union[datetime, Time]) -> Tuple[datetime, datetime]:
        """
        Convert UTC time to Chile local time and Korean time.
        
        Args:
            utc_time: Datetime in UTC or astropy Time object
            
        Returns:
            Tuple containing (chile_time, korea_time)
        """
        try:
            # Handle astropy Time objects
            if isinstance(utc_time, Time):
                utc_datetime = utc_time.datetime
            else:
                utc_datetime = utc_time
                
            # Ensure UTC time has timezone info
            if utc_datetime.tzinfo is None:
                utc_datetime = pytz.utc.localize(utc_datetime)
            elif utc_datetime.tzinfo != pytz.utc:
                # Convert to UTC if it's in a different timezone
                utc_datetime = utc_datetime.astimezone(pytz.utc)
                
            # Convert to Chile and Korea times
            chile_time = utc_datetime.astimezone(self.chile_tz)
            korea_time = utc_datetime.astimezone(self.korea_tz)
            
            self.logger.debug(f"Time conversion - UTC: {utc_datetime}, CLT: {chile_time}, KST: {korea_time}")
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
        Analyze visibility status based on staralt data and determine observability.
        
        Args:
            staralt_data_dict: Dictionary containing staralt analysis results
            
        Returns:
            Dictionary with visibility status and details
        """
        # Initialize result structure
        result = {
            "status": "unknown",
            "condition": "Analysis pending", 
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
            
            # Ensure we're using the correct current time
            if now_datetime is None:
                from astropy.time import Time
                now_datetime = Time.now().datetime
                self.logger.warning("now_datetime was None, using current UTC time as fallback")
                
            # Extract sunset/sunrise for debugging
            sunset_night = tonight.get("sunset_night")
            sunrise_night = tonight.get("sunrise_night")
            
            # Enhanced debugging
            self.logger.debug(f"=== VISIBILITY ANALYSIS DEBUG ===")
            self.logger.debug(f"Current time: {now_datetime}")
            self.logger.debug(f"Current date: {now_datetime.date() if now_datetime else 'None'}")
            self.logger.debug(f"Tonight's sunset: {sunset_night}")
            self.logger.debug(f"Tonight's sunrise: {sunrise_night}")
            
            # Validate input data
            if not target_times or not target_alts:
                self.logger.error("Missing target times or altitudes data")
                result["status"] = "error"
                result["condition"] = "Missing observational data"
                return result
            
            # Convert target_times to datetime objects if they aren't already
            target_times_dt = []
            for t in target_times:
                if isinstance(t, str):
                    target_times_dt.append(datetime.fromisoformat(t))
                elif hasattr(t, 'datetime'):
                    target_times_dt.append(t.datetime)
                else:
                    target_times_dt.append(t)
            
            # Find current time index
            now_idx = None
            min_time_diff = float('inf')
            for i, t in enumerate(target_times_dt):
                time_diff = abs((t - now_datetime).total_seconds())
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    now_idx = i
            
            if now_idx is None:
                self.logger.error("Could not find current time index")
                result["status"] = "error"
                result["condition"] = "Time indexing failed"
                return result
            
            # Update current conditions
            result["current_altitude"] = target_alts[now_idx] if now_idx < len(target_alts) else 0
            result["current_moon_separation"] = target_moonsep[now_idx] if now_idx < len(target_moonsep) else 0
            
            # Find observable periods (green color indicates observable)
            observable_indices = [i for i, color in enumerate(color_target) if color == 'green']
            
            self.logger.debug(f"Found {len(observable_indices)} observable periods")
            
            # Check if currently observable
            currently_observable = now_idx in observable_indices
            
            if currently_observable:
                # Case 1: Observable now
                return self._handle_observable_now(result, observable_indices, target_times_dt, now_datetime)
            else:
                # Case 2: Observable later tonight
                # FIXED: Remove the extra arguments that were causing the error
                return self._handle_observable_later(result, observable_indices, target_times_dt, 
                                                    target_alts, target_moonsep, now_idx)
        
        except Exception as e:
            self.logger.error(f"Error in visibility analysis: {e}", exc_info=True)
            result["status"] = "error"
            result["condition"] = f"Analysis failed: {str(e)}"
            result["message"] = "Could not analyze visibility"
            result["recommendation"] = "Manual observation planning required"
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
                                target_times_dt: List[datetime], target_alts: List[float], 
                                target_moonsep: List[float], now_idx: int) -> Dict[str, Any]:
        """
        Handle Case 2: Observable later tonight
        
        Args:
            result: Dictionary to populate with visibility results
            observable_indices: List of indices where target is observable
            target_times_dt: List of datetime objects for target times
            target_alts: List of target altitudes (not used but kept for compatibility)
            target_moonsep: List of moon separations (not used but kept for compatibility)
            now_idx: Index of current time in the arrays
            
        Returns:
            Updated result dictionary with visibility information
        """
        self.logger.info("Processing target observable later tonight")
        
        # Get current datetime from the target_times_dt array
        now_datetime = target_times_dt[now_idx]
        
        result["status"] = "observable_later"
        
        # Find next observable window
        future_indices = [i for i in observable_indices if i > now_idx]
        if not future_indices:
            # This shouldn't happen if we're in this case, but handle it gracefully
            result["status"] = "not_observable"
            result["condition"] = "No future observable periods tonight"
            result["message"] = "Target becomes unobservable for the rest of the night"
            result["recommendation"] = "Consider tomorrow's visibility"
            return result
        
        start_idx = min(future_indices)
        end_idx = max(observable_indices)
        
        result["observable_start"] = target_times_dt[start_idx]
        result["observable_end"] = target_times_dt[end_idx]
        result["observable_hours"] = (result["observable_end"] - result["observable_start"]).total_seconds() / 3600
        
        # Calculate precise time until observable
        time_until_observable = (result["observable_start"] - now_datetime).total_seconds() / 3600
        result["hours_until_observable"] = max(0, time_until_observable)
        
        # Calculate minutes for more precision when less than 1 hour
        if result["hours_until_observable"] < 1:
            minutes_until = (result["observable_start"] - now_datetime).total_seconds() / 60
            result["minutes_until_observable"] = max(0, minutes_until)
            self.logger.info(f"Observable in {minutes_until:.0f} minutes")
        else:
            self.logger.info(f"Observable in {time_until_observable:.1f} hours")
        
        # Enhanced condition setting
        if time_until_observable < 0.5:  # Less than 30 minutes
            result["condition"] = "Observable Very Soon"
            result["urgency"] = "high"
            result["recommendation"] = "Prepare for immediate observations"
        elif time_until_observable < 1:  # Less than 1 hour
            result["condition"] = "Observable Soon"
            result["urgency"] = "medium"
            result["recommendation"] = "Prepare observation setup"
        elif time_until_observable < 3:  # Less than 3 hours
            result["condition"] = "Observable Later Tonight"
            result["urgency"] = "low"
            result["recommendation"] = "Plan observations for later"
        else:
            result["condition"] = "Observable Much Later"
            result["urgency"] = "low"
            result["recommendation"] = "Consider if worth waiting"
        
        result["message"] = f"Observable in {result['hours_until_observable']:.1f} hours"
        
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

    def _generate_today_visibility(self, ra: float, dec: float, grb_name: str, 
                                minalt: float, minmoonsep: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Generate today's visibility analysis for the target.
        
        Args:
            ra: Right Ascension in degrees
            dec: Declination in degrees  
            grb_name: Name of the GRB
            minalt: Minimum altitude in degrees
            minmoonsep: Minimum moon separation in degrees
            
        Returns:
            Tuple of (visibility_dict, data_dict)
        """
        try:
            self.logger.info(f"Generating today's visibility for {grb_name} at RA={ra:.2f}, DEC={dec:.2f}")
            
            # Get current time
            current_time = Time.now()
            current_datetime = current_time.datetime
            current_hour = current_datetime.hour
            
            # Determine appropriate night reference time
            # Logic: 
            # - If current time is before noon (12:00), we're still in "last night"
            # - If current time is after noon, we're planning for "tonight"
            if current_hour < 12:
                # Early morning - we're still in last night's observing session
                # Use yesterday's date for night calculation
                night_reference_time = current_time - 1*u.day
                self.logger.debug(f"Early morning ({current_hour:02d}:xx) - using yesterday's night definition")
            else:
                # Afternoon/evening - planning for tonight's observing session
                night_reference_time = current_time
                self.logger.debug(f"Afternoon/evening ({current_hour:02d}:xx) - using today's night definition")
            
            self.logger.debug(f"Night reference time: {night_reference_time.datetime}")

            # Calculate visibility for the appropriate night
            self.staralt.set_target(
                ra=ra,
                dec=dec,
                objname=grb_name,
                utctime=night_reference_time,  # Use the adjusted reference time
                target_minalt=minalt,
                target_minmoonsep=minmoonsep
            )
            
            # Validate night calculation
            data_dict = self.staralt.data_dict
            tonight = data_dict.get("tonight", {})
            sunset_night = tonight.get("sunset_night")
            sunrise_night = tonight.get("sunrise_night")
            
            if sunset_night and sunrise_night:
                # Handle both datetime and astropy.Time objects correctly
                if hasattr(sunset_night, 'datetime'):
                    sunset_dt = sunset_night.datetime
                    sunrise_dt = sunrise_night.datetime
                else:
                    # Already datetime objects
                    sunset_dt = sunset_night
                    sunrise_dt = sunrise_night
                
                self.logger.debug(f"Calculated night: {sunset_dt} to {sunrise_dt}")
                
                # Check if current time falls within the calculated night
                if sunset_dt <= current_datetime <= sunrise_dt:
                    self.logger.info("Current time is within the calculated observing night")
                elif current_datetime < sunset_dt:
                    hours_until_night = (sunset_dt - current_datetime).total_seconds() / 3600
                    self.logger.info(f"Current time is {hours_until_night:.1f}h before tonight's observing window")
                else:
                    self.logger.info("Current time is after tonight's observing window")
            
            today_visibility = self._analyze_visibility_status(self.staralt.data_dict)
            self.logger.info(f"Today's night visibility analysis complete: {today_visibility.get('status', 'unknown')}")
            
            return today_visibility, data_dict
            
        except Exception as e:
            self.logger.error(f"Error generating today's visibility: {e}")
            error_visibility = {
                "status": "error",
                "condition": f"Analysis failed: {str(e)}",
                "message": "Could not analyze visibility",
                "recommendation": "Manual observation planning required"
            }
            return error_visibility, {}

    def _generate_tomorrow_visibility(self, ra: float, dec: float, grb_name: str, minalt: float, minmoonsep: float) -> Tuple[Dict[str, Any], Any]:
        """Generate visibility data for tomorrow night"""
        self.logger.info(f"Generating tomorrow's visibility for {grb_name}")
        
        try:
            # Calculate tomorrow's date (next observing night)
            tomorrow = Time.now() + 1*u.day
            self.logger.debug(f"Tomorrow's date: {tomorrow.datetime.strftime('%Y-%m-%d')}")
            
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
            
            # Add observation limit warning if present
            if visibility_info.get("observation_limit_warning", False):
                warning_msg = visibility_info.get("warning_message", "")
                sections.append(f"> {warning_msg}")
                self.logger.debug("Added observation limit warning to visibility message")
            
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