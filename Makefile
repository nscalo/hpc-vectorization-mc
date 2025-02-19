CXX=icpc
CXXFLAGS=-QaxMIC-AVX512 -Qopenmp -Qopenmp-simd -Qmkl
OPTRPT=-Qopt-report=5
# CXXLIB=-cxxlib="C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\include"

default : app

distribution.o : distribution.cc distribution.h
	icpc -c ${OPTRPT} ${CXXFLAGS} -o "$@" "$<"

diffusion.o : diffusion.cc distribution.o
	icpc -c ${OPTRPT} ${CXXFLAGS} -o "$@" "$<" distribution.obj

app : main.cc diffusion.o distribution.o
	icpc ${OPTRPT} ${CXXFLAGS} -o "$@" "$<" diffusion.obj distribution.obj

all: app

queue: all
	TIMEIT -f .\timeit.dat -m 0x1 app.exe

clean :
	rm app diffusion.o distribution.o *.optrpt diffusion.obj distribution.obj