# compiler
CC=g++
# linker
LD=$(CC)
# optimisation
OPT=-ggdb
# warnings
WARN=-Wall -Wextra
# standards
STD=c++17
# pthread
PTHREAD=-pthread
# PTHREAD=

TARGET = blur

CCFLAGS = $(WARN) $(PTHREAD) -std="$(STD)"  $(OPT) -pipe  -Iboost `pkg-config --cflags OpenCL`
LDFLAGS = $(PTHREAD) `pkg-config --libs  OpenCL` -ljpeg

SRCS = $(wildcard *.cpp)
OBJECTS = $(patsubst %.cpp, %.o, $(SRCS))

.PHONY: default all clean

default: $(TARGET)

all: default

%.o: %.cpp
	$(CC) $(CCFLAGS) -c $< -o $@

.PRECIOUS: $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
	$(LD) $(OBJECTS) $(LDFLAGS) -o $@

clean:
	$(RM) *.o
	$(RM) $(TARGET)
