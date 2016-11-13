import time
import sys
import os
import logging
import logging_colorer
logging_colorer.init_logging()

DEBUG_SCRIPT = "debug_script.py"

try:
    last_time = os.path.getmtime(DEBUG_SCRIPT)
except FileNotFoundError:
    logging.error("No debug file")
    last_time = 0

def i_have_it_all(**kwargs):
    print(i)

for i in range(int(1e10)):
    try:
        if os.path.getmtime(DEBUG_SCRIPT) >= last_time:
            last_time = time.time()

            with open(DEBUG_SCRIPT, 'r') as f:
                script=f.read()
                logging.info("Running debug file")
                exec(script)

    except SyntaxError as e:
        logging.error("Badly written script!")
        print(e)
    except FileNotFoundError:
        logging.error("No debug file")

    print("%d" % i, end="\r")

    i_have_it_all(**locals())

