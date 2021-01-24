#include ../../../mkfiles/make.inc
#include ../../../mkfiles/Make.common
SWIG = /usr/bin/swig 
PYINC = /scratch/Anaconda3-4.3.1-Linux-x86_64/include/python3.6m
GCC = /usr/bin/gcc
MYLIB = /home/yanglu/move37/prototype/Floorplanner/src
MYINC = /home/yanglu/move37/prototype/Floorplanner/src

INCFLAG = -I$(PYINC)
INCFLAG += -I$(MYINC)


CXXFLAGS = -O3 -Wall -std=c++11 -fPIC

all: fastseqpair 

fastseqpair:
	@echo "generating swig wrapped lib fastseqpair"
	@rm -f fastseqpair.py *.h *.cpp
	@$(SWIG) -c++ -python $(INCFLAG) -o fastseqpair_wrap.cpp fastseqpair.i
	@echo "CXXFLAGS = " $(CXXFLAGS), "OPT_LEVEL = " $(OPT_LEVEL)  #
	@$(GCC) -shared $(CXXFLAGS) fastseqpair_wrap.cpp -o _fastseqpair.so -Wl,-rpath,$(MYLIB) -L$(MYLIB) -lseqpair $(INCFLAG)

.PHONY: fastseqpair 
#include ../../../mkfiles/Make.rules
