###################################

# EpilepsyLAB

# Package: io
# File: user
# Description: Procedures to read and write to the console.

# Contributors: João Saraiva
# Last update: 01/11/2021

###################################

from colorama import Back


def log(message:str, attention=0):
    """
    Prints a message as a program log.
    :param message: The message to print.
    :param attention: The level of attention: (0) Information, (1) Warning, (2) Error and exits with -1 code.
    """
    attention_levels = (Back.GREEN + 'INFO', Back.YELLOW + 'WARN', Back.RED + 'ERROR')
    print("[{}]".format(attention_levels[attention]), end=' ')
    print(message)
    if attention == 2:
        exit(-1)


def query(question:str, yesorno=False):
    """
    Queries the user in the console.
    :param question: The question to print.
    :param yesorno: Pass as True if it is a yes or no question.
    :return: The typed answer as a string; or True/False is yesorno is True.
    """
    if yesorno:
        answer = input(question + ' [y/n]')
        return True if answer.lower() in ('y', 'yes') else False
    else:
        return input(question)