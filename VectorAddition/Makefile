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


OCLCCFLAGS = -I /opt/AMDAPPSDK-3.0/include/
OCLLDFLAGS = -L /opt/AMDAPPSDK-3.0/lib/x86_64/ -lOpenCL

CCFLAGS = $(WARN) $(PTHREAD) -std="$(STD)"  -stdlib=libstdc++  $(OPT) $(OCLCCFLAGS)
LDFLAGS = $(PTHREAD)  $(OCLLDFLAGS)

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
