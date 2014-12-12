
include env.mk

C_SRC_OUTPUT = ../priv/eblas.so

UNAME_SYS := $(shell uname -s)
ifeq ($(UNAME_SYS), Darwin)
	CC ?= cc
	CFLAGS ?= -O3 -std=c99 -arch x86_64 -flat_namespace -undefined suppress -finline-functions -Wall -Wmissing-prototypes
else ifeq ($(UNAME_SYS), FreeBSD)
	CC ?= cc
	CFLAGS ?= -O3 -std=c99 -finline-functions -Wall -Wmissing-prototypes
else ifeq ($(UNAME_SYS), Linux)
	CC ?= gcc
	CFLAGS ?= -O3 -std=c99 -finline-functions -Wall -Wmissing-prototypes
endif

$(C_SRC_OUTPUT): *.c 
	$(CC) $(CFLAGS) *.c -fPIC -shared -o $(C_SRC_OUTPUT) -I $(ERTS_INCLUDE_DIR) \
		$(C_SRC_OPTS) -L/usr/lib -l:libcblas.so.3 -l:libatlas.so.3



