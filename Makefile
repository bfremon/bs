python3 = /usr/bin/python3

.PHONY: all

all:
	for f in *.py;  do echo '\nTesting '$$f; $(python3) $$f; done
