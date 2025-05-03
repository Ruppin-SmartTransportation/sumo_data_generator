import datetime
from termcolor import colored

class Logger:
    """ Handles logging to both the console and a log file with timestamps, colors, and source info. """

    def __init__(self, log_file_path="main/simulation_log.log"):
        """ Initialize the logger with a log file path. """
        self.log_file = open(log_file_path, "w", encoding="utf-8")
        self.log("Simulation log file initialized.", "INFO", "cyan", 
                 class_name="Logger", function_name="__init__", print_to_console=True)
        

    def log(self, message, level="INFO", color=None, class_name="UNKNOWN", function_name="UNKNOWN", print_to_console=False):
        """
        Logs a message to the console and the log file with timestamp, level, class name, and function name.

        :param message: The message to log.
        :param level: The log level (INFO, WARNING, ERROR, etc.).
        :param color: The color for the console output.
        :param class_name: The name of the class calling the log.
        :param function_name: The name of the function calling the log.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] [{class_name}::{function_name}] {message}"

        # Print to console
        if print_to_console:
            if color:
                print(colored(log_entry, color))
            else:
                print(log_entry)

        # Write to log file
        self.log_file.write(log_entry + "\n")

    def close(self):
        """ Closes the log file. """
        self.log_file.close()
