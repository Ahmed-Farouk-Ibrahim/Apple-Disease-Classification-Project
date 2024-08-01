import os
import sys
import logging

# Define the logging format:  including the timestamp, logging level, module name/Running_File, and the message itself.
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# Directory where log files will be stored:
log_dir = "logs"

# Constructs the full file path for the log file (running_logs.log).
log_filepath = os.path.join(log_dir, "running_logs.log")

# Create the log directory if it doesn't exist: 
os.makedirs(log_dir, exist_ok=True)

# Configure the logging settings
logging.basicConfig(
    # Set the logging level to INFO, which means that all messages with the level INFO and above will be logged.
    level=logging.INFO,  
    # Specify the format for log messages
    format=logging_str,  
    # Adds two handlers:
    handlers=[ 
        # Log messages to a file
        logging.FileHandler(log_filepath), 
        # Also output log messages to the console 
        logging.StreamHandler(sys.stdout)  
    ]
)

# Creates a logger object named "cnnProjectLogger" which can be used throughout the project to log messages.
logger = logging.getLogger("cnnProjectLogger")

# Check logging is working:
if __name__=='__main__':
    logging.info("Logging is working")
