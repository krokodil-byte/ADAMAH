CC = gcc
CFLAGS = -O3 -Wall
LDFLAGS = -lvulkan -lm

all: libadamah.so test

libadamah.so: adamah.c adamah.h
	$(CC) -shared -fPIC $(CFLAGS) adamah.c -o $@ $(LDFLAGS)

test: adamah.c test.c adamah.h
	$(CC) $(CFLAGS) adamah.c test.c -o $@ $(LDFLAGS)
	./test

clean:
	rm -f libadamah.so test *.bin

.PHONY: all clean
