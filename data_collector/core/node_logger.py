import datetime
from termcolor import colored

class NodesLogger:
    """ Handles logging specifically for nodes (static and dynamic) to a separate file. """

    def __init__(self, log_file_path="main/nodes_log.log"):
        """ Initialize the nodes logger. """
        self.log_file = open(log_file_path, "w", encoding="utf-8")
        self.log("Nodes log file initialized.", "INFO", 
                 class_name="NodesLogger", function_name="__init__", print_to_console=True)

    def log(self, message, level="INFO", class_name="UNKNOWN", function_name="UNKNOWN", print_to_console=False):
        """
        Logs a message to the nodes log file.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] [{class_name}::{function_name}] {message}"

        # Write to log file
        self.log_file.write(log_entry + "\n")

        # Print to console
        if print_to_console:
            print(log_entry)
        
    def close(self):
        """ Closes the nodes log file. """
        self.log_file.close()