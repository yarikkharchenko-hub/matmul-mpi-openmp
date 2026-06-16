/*
 * matmul_hybrid.c — Гібридна реалізація MPI + OpenMP
 * Модель MPI_THREAD_FUNNELED: тільки головна нить викликає MPI
 * Оптимальна конфігурація: 4 MPI-процеси × 3 OpenMP-потоки = 12 ядер
 *
 * Компіляція: mpicc -O3 -fopenmp -o matmul_hybrid matmul_hybrid.c -lm
 * Запуск:     mpirun -np 4 ./matmul_hybrid 4096 3
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

typedef struct {
    double t_bcast;
    double t_scatter;
    double t_compute;
    double t_gather;
    double t_barrier;
    double t_total;
    int    omp_threads;
} HybridOverhead;

static double *alloc_matrix(int n) {
    double *m = (double *)malloc((size_t)n * n * sizeof(double));
    if (!m) { MPI_Abort(MPI_COMM_WORLD, 1); }
    return m;
}

/* NUMA-коректна ініціалізація: кожен потік ініціалізує свій блок */
static void init_parallel(double *m, int rows, int cols,
                           int row_offset, int nthreads) {
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            m[i*cols+j] = (row_offset + i + j + 1) * 0.1;
}

static void transpose_parallel(const double *B, double *BT,
                                 int n, int nthreads) {
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            BT[j*n+i] = B[i*n+j];
}

/*
 * Локальне множення OpenMP: rows рядків A × повна BT
 * schedule(static) — рівномірний розподіл, оптимально для матриць
 */
static void local_matmul_omp(const double *Aloc, const double *BT,
                               double *Cloc, int rows, int n, int nthreads) {
    memset(Cloc, 0, (size_t)rows * n * sizeof(double));

    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int i = 0; i < rows; i++) {
        const double *arow = Aloc + i*n;
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            const double *btrow = BT + j*n;
            for (int k = 0; k < n; k++)
                sum += arow[k] * btrow[k];
            Cloc[i*n+j] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    int provided;
    /* MPI_THREAD_FUNNELED: тільки головна нить (rank 0) викликає MPI */
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0 && provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "Warning: MPI_THREAD_FUNNELED не підтримується, "
                        "provided=%d\n", provided);
    }

    int n        = (argc > 1) ? atoi(argv[1]) : 4096;
    int nthreads = (argc > 2) ? atoi(argv[2]) : 3;

    /* Обмеження потоків кількістю доступних ядер */
    int max_threads = omp_get_max_threads();
    if (nthreads > max_threads) nthreads = max_threads;

    int rows = n / size;

    HybridOverhead oh = {0};
    oh.omp_threads = nthreads;

    /* ====== Пам'ять ====== */
    double *A    = NULL;
    double *C    = NULL;
    double *B    = alloc_matrix(n);
    double *BT   = alloc_matrix(n);
    double *Aloc = (double *)malloc((size_t)rows * n * sizeof(double));
    double *Cloc = (double *)malloc((size_t)rows * n * sizeof(double));

    if (rank == 0) {
        A = alloc_matrix(n);
        C = alloc_matrix(n);
        /* Rank 0 ініціалізує з OpenMP (NUMA node 0) */
        init_parallel(A, n, n, 0, nthreads);
        init_parallel(B, n, n, 1, nthreads);
    }

    /* Синхронізація */
    double tb0 = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    oh.t_barrier = MPI_Wtime() - tb0;

    /* ====== MPI_Bcast(B): головна нить надсилає ====== */
    tb0 = MPI_Wtime();
    MPI_Bcast(B, n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    oh.t_bcast = MPI_Wtime() - tb0;

    /* Транспонування — паралельно всередині процесу */
    transpose_parallel(B, BT, n, nthreads);

    /* ====== MPI_Scatter(A) ====== */
    double ts0 = MPI_Wtime();
    MPI_Scatter(A, rows*n, MPI_DOUBLE,
                Aloc, rows*n, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    oh.t_scatter = MPI_Wtime() - ts0;

    /* Кожен MPI-процес ініціалізує свій Aloc через OpenMP (first-touch) */
    /* Алок вже заповнений через Scatter — це демонстраційна ініц. */

    /* ====== Локальне множення OpenMP ====== */
    MPI_Barrier(MPI_COMM_WORLD);
    double tc0 = MPI_Wtime();
    local_matmul_omp(Aloc, BT, Cloc, rows, n, nthreads);
    oh.t_compute = MPI_Wtime() - tc0;

    /* ====== MPI_Gather(C) ====== */
    double tg0 = MPI_Wtime();
    MPI_Gather(Cloc, rows*n, MPI_DOUBLE,
               C,    rows*n, MPI_DOUBLE,
               0, MPI_COMM_WORLD);
    oh.t_gather = MPI_Wtime() - tg0;

    oh.t_total = oh.t_bcast + oh.t_scatter + oh.t_compute + oh.t_gather + oh.t_barrier;

    /* Зведення max через усі процесори */
    double max_compute, max_bcast, max_scatter, max_gather, max_total;
    MPI_Reduce(&oh.t_compute, &max_compute, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&oh.t_bcast,   &max_bcast,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&oh.t_scatter, &max_scatter, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&oh.t_gather,  &max_gather,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&oh.t_total,   &max_total,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double overhead = max_bcast + max_scatter + max_gather + oh.t_barrier;

        printf("=== Гібрид MPI+OpenMP: n=%d, %d×%d ===\n",
               n, size, nthreads);
        printf("Конфігурація: %d MPI-процеси × %d OpenMP-потоки = %d ядер\n\n",
               size, nthreads, size*nthreads);

        printf("┌────────────────────────┬──────────────┬──────────┐\n");
        printf("│ Компонент              │ Час (мс)     │ %%        │\n");
        printf("├────────────────────────┼──────────────┼──────────┤\n");
        printf("│ MPI_Bcast(B)           │ %10.2f   │ %6.2f%%  │\n",
               max_bcast*1000,   max_bcast/max_total*100);
        printf("│ MPI_Scatter(A)         │ %10.2f   │ %6.2f%%  │\n",
               max_scatter*1000, max_scatter/max_total*100);
        printf("│ MPI_Gather(C)          │ %10.2f   │ %6.2f%%  │\n",
               max_gather*1000,  max_gather/max_total*100);
        printf("│ 3. Синхронізація (Barrier)       │ %10.2f   │ %6.2f%%  │\n",
               oh.t_barrier*1000, oh.t_barrier/max_total*100);
        printf("│ Обчислення (OMP×%d)   │ %10.2f   │ %6.2f%%  │\n",
               nthreads, max_compute*1000, max_compute/max_total*100);
        printf("├────────────────────────┼──────────────┼──────────┤\n");
        printf("│ Всього НВ              │ %10.2f   │ %6.2f%%  │\n",
               overhead*1000, overhead/max_total*100);
        printf("│ Загальний час          │ %10.2f   │ 100.00%%  │\n",
               max_total*1000);
        printf("└────────────────────────┴──────────────┴──────────┘\n");

        double flops = 2.0 * n * n * n;
        printf("\nGFLOP/s: %.2f\n", flops / max_total / 1e9);
        printf("MPI provided level: %d (need %d)\n", provided, MPI_THREAD_FUNNELED);

        free(A); free(C);
    }

    free(B); free(BT); free(Aloc); free(Cloc);
    MPI_Finalize();
    return 0;
}
