# executable name
TARGET = VecAdd.out

# compiler
CC=clang++
# linker
LD=$(CC)
# optimisation
OPT=-ggdb
# warnings
WARN=-Wall -Wextra
# standards
STD=c++14
# pthread
PTHREAD=-pthread


CCFLAGS = $(WARN) $(PTHREAD) -std="$(STD)"  -stdlib=libstdc++  $(OPT) -I /opt/AMDAPPSDK-3.0/include/
LDFLAGS = $(PTHREAD) -L /opt/AMDAPPSDK-3.0/lib/x86_64/ -lOpenCL

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
