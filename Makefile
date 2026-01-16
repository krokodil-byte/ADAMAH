CC = gcc
CFLAGS = -O3 -Wall
LDFLAGS = -lvulkan -lm

all: test

test: adamah.c test.c adamah.h
	$(CC) $(CFLAGS) adamah.c test.c -o test_adamah $(LDFLAGS)
	./test_adamah

clean:
	rm -f test_adamah

.PHONY: all test clean
