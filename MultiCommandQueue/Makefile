# compiler
CC=g++
# linker
LD=$(CC)
# optimisation
OPT=-Ofast # -ggdb
# warnings
WARN=-Wall -Wextra
# vars
VARS = -D OPENCL_1
# standards
STD=c++17
# pthread
PTHREAD=-pthread
# PTHREAD=

TARGET = multi_cq

CCFLAGS = $(WARN) $(PTHREAD) -std="$(STD)" $(VARS) $(OPT) -pipe  -Iboost `pkg-config --cflags OpenCL` # `Magick++-config --cppflags --cxxflags` # -cl-std=CL2.0
LDFLAGS = $(PTHREAD) `pkg-config --libs  OpenCL` # `Magick++-config --ldflags --libs`  # -export-dynamic

SRCS = $(wildcard *.cc)
OBJECTS = $(patsubst %.cc, %.o, $(SRCS))

.PHONY: default all clean

default: $(TARGET) Makefile

all: default

%.o: %.cc
	$(CC) $(CCFLAGS) -c $< -o $@

.PRECIOUS: $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
	$(LD) $(OBJECTS) $(LDFLAGS) -o $@

clean:
	$(RM) *.o
	$(RM) $(TARGET)
