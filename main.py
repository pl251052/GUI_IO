
from login import loginscreen  # Importing login screen function from login.py
from createaccount import *  # Importing everything from createaccount.py
from gui import *  # Importing everything from gui.py
import os  # Importing file managing library, which allows for editing and reading other files
from logdefiner import get_log_name

filename = get_log_name()
if os.path.exists(filename):
    pass
else:
    f = open(filename, "w")

# Builds Login Screen - has create account and make gui inside of it
loginscreen()