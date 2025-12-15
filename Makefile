.PHONY: all clean

all: src/Makefile
	cd src; make PREFETCH=1 PARLAY=1; cd ..

clean:
	rm src/*.o
