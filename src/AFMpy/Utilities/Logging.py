import logging
import random

__all__ = ['make_module_logger', 'get_module_logger', 'set_logging_level', 'set_logfile', 'configure_module_logging', 'print_Ioan_quote']

def make_module_logger(module_name):
    '''
    Creates a logger for a module with a base configuration. This function should only be used in modules, as
    it may create loggers that are node desired.

    Parameters:
        module_name (module):
            The module name to create the logger for.

    Returns (logging.Logger):
        The logger for the given module.

    '''
    # Create the logger for the module
    logger = logging.getLogger(module_name)
    
    # Set the initial logging level to WARNING
    logger.setLevel(logging.WARNING)

    # Create the logging handler. This handler will print to the console by default.
    handler = logging.StreamHandler()

    # Create the formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Clear the handlers and add the new handler
    logger.handlers.clear()
    logger.addHandler(handler)

    return logger

def get_module_logger(module_name):
    '''
    Returns the logger for the given module name. If the logger does not exist, it will be created.
    Be careful when using this function outside of modules as it may create loggers that are not desired.

    Parameters:
        module_name (str):
            The name of the module to get the logger for.

    '''
    return logging.getLogger(module_name)

def set_logging_level(logger, level_str):
    '''
    Set the Logging level for an existing logger based upon a string.
    
    Parameters:
        logger (logging.Logger):
            The logger to set the logging level for.
        level_str (str):
            The logging level to set. Must be one of the following:
                'debug', 'info', 'warning', 'error', 'critical'
    
    Raises:
        ValueError:
            If the provided level string is not valid.            

    Returns (None):
        None
    '''
    # Handle case sensitivity
    level_str = level_str.lower()  
    
    level_mapping = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    
    # Check if the provided level string is valid
    if level_str not in level_mapping:
        raise ValueError(f"Invalid logging level: {level_str}")
    
    level = level_mapping[level_str]

    logger.setLevel(level)

def set_logfile(logger, logfile):
    '''
    Sets the logging for an existing logger to a file location.

    Parameters:
        logger (logging.Logger):
            The logger to set the logging file for.
        logfile (str):
            The path to the logging file.
    
    Returns (None): 
        None
    '''
    # Create the file handler
    file_handler = logging.FileHandler(logfile, mode='a')

    # Create the formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Set the logging level to that of the logger
    file_handler.setLevel(logger.level)
    
    # Clear the handlers and add the new handler
    logger.handlers.clear()
    logger.addHandler(file_handler)

def configure_module_logging(module_name, enable=True, level_str='warning', logfile=None):
    '''
    Configures the logging for a given module.

    Parameters:
        module (module):
            The module name of the logger to configure

        enable (bool):
            Whether or not to enable logging.

        level_str (str):
            The logging level to set. Must be one of the following:
                'debug', 'info', 'warning', 'error', 'critical'

        logfile (str):
            The path to the logging file.
    
    Returns (None):
        None    
    '''
    logger = get_module_logger(module_name)

    # Enable All Logging
    if enable:
        logger.disabled = False
    else:
        logger.disabled = True
        return
    
    # Set the logging level
    set_logging_level(logger, level_str)

    # Turn off logger propogation to ancesotr loggers
    logger.propagate = False
    
    # Set the log file if specified
    if logfile:
        set_logfile(logger, logfile)


def get_Ioan_quote():
    '''
    Returns a random quote from Dr. Ioan Kosztin.

    Parameters:
        None

    Returns (str):
        A random quote from Dr. Ioan Kosztin.
    '''
    ioan_quotes = ['Remember, there are no silver bullets... - Ioan Kosztin',
                'Remember, there\'s no free lunch... - Ioan Kosztin',
                'Remember, garbage in, garbage out... - Ioan Kosztin',
                'Registration or Clustering; The Chicken or the Egg... - Ioan Kosztin',
                'Your method is like a pencil... You can sharpen it all day long, but will you ever write with it? - Ioan Kosztin',
                ]
    
    return random.choice(ioan_quotes)

def log_Ioan_quote(logger: logging.Logger):
    '''
    Logs a random quote from Dr. Ioan Kosztin.

    Parameters:
        logger (logging.Logger):
            The logger to log the quote to.
    
    Returns (None):
        None
    '''
    quote = get_Ioan_quote()
    logger.info(quote)

def print_Ioan_quote():
    '''
    Prints a random quote from Dr. Ioan Kosztin.

    Parameters:
        None
    
    Returns (None):
        None
    '''
    quote = get_Ioan_quote()
    print(quote)