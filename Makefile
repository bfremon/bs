python3 = /usr/bin/python3
rm = /bin/rm -fr

.PHONY: all clean

all:
	for f in *.py;  do echo '\nTesting '$$f; $(python3) $$f; done

clean:
	$(rm) -fr bench/ __pycache__/ fit_dat.csv stats.csv *.png
