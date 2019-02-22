import sys
import time
from math import floor,log10

# prints a progress bar in the terminal
def progress(current, total):
    bar_length = 60
    filled_length = int(round(bar_length * current / float(total)))

    percents = round(100.0 * current / float(total), 1)
    bar = '=' * max(filled_length-1,0) + '>' * min(filled_length,1) + '.' * (bar_length - filled_length)
	
    #  in command line is constantly being replaced output 
    sys.stdout.write(f"\r[{bar}] {_fill_number(percents,100)}% {_fill_number(current,total)}/{total}")
    sys.stdout.flush()

# fills the smaller number with spaces, if this is needed
def _fill_number(num,biggest):
    return ( floor(log10(biggest))- floor(log10(max(num,1))) ) * " " + str(num)

if __name__ == '__main__':
	# this code is only for testing purposes
	for i in range(300):
		time.sleep(0.1)
		progress(i+1,300)
	sys.stdout.write("\n")

