python3 = /usr/bin/python3

.PHONY: all

all:
	for f in *.py;  do $(python3) $$f; done
