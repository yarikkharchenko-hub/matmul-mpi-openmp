# Makefile для паралельного множення матриць
# Харченко Я.О. — бакалаврська робота

CC      = gcc
MPICC   = mpicc
CFLAGS  = -O3 -Wall -Wextra
OMPFLAGS= -fopenmp
LIBS    = -lm

.PHONY: all seq omp mpi hybrid clean run-all benchmark

all: matmul_seq matmul_omp matmul_mpi matmul_hybrid

matmul_seq: matmul_seq.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

matmul_omp: matmul_omp.c
	$(CC) $(CFLAGS) $(OMPFLAGS) -o $@ $< $(LIBS)

matmul_mpi: matmul_mpi.c
	$(MPICC) $(CFLAGS) -o $@ $< $(LIBS)

matmul_hybrid: matmul_hybrid.c
	$(MPICC) $(CFLAGS) $(OMPFLAGS) -o $@ $< $(LIBS)

# Швидкий тест (n=512)
test: all
	@echo "\n--- Послідовний ---"
	./matmul_seq 512
	@echo "\n--- OpenMP (4 потоки) ---"
	OMP_NUM_THREADS=4 ./matmul_omp 512 4
	@echo "\n--- MPI (4 процеси) ---"
	mpirun -np 4 ./matmul_mpi 512
	@echo "\n--- Гібрид 2×2 ---"
	mpirun -np 2 ./matmul_hybrid 512 2

# Повний бенчмарк (n=1024)
benchmark: all
	@echo "\n=== BENCHMARK n=1024 ==="
	@echo "\n--- Послідовний ---"
	./matmul_seq 1024
	@for t in 2 4 8; do \
		echo "\n--- OpenMP t=$$t ---"; \
		OMP_NUM_THREADS=$$t ./matmul_omp 1024 $$t; \
	done
	@for p in 2 4 8; do \
		echo "\n--- MPI p=$$p ---"; \
		mpirun -np $$p ./matmul_mpi 1024; \
	done
	@echo "\n--- Гібрид 4×3 ---"
	mpirun -np 4 ./matmul_hybrid 1024 3

clean:
	rm -f matmul_seq matmul_omp matmul_mpi matmul_hybrid
