/*
 * matmul_mpi.c — Паралельне множення матриць засобами MPI
 * Рядкова декомпозиція + профілювання накладних витрат (6 категорій)
 *
 * Компіляція: mpicc -O3 -o matmul_mpi matmul_mpi.c -lm
 * Запуск:     mpirun -np 8 ./matmul_mpi 1024
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

typedef struct {
    double t_init;        /* 1. Ініціалізація (MPI_Init/Finalize) */
    double t_bcast;       /* 2. Комунікації: MPI_Bcast(B) */
    double t_scatter;     /* 2. Комунікації: MPI_Scatter(A) */
    double t_gather;      /* 2. Комунікації: MPI_Gather(C) */
    double t_barrier;     /* 3. Синхронізація: MPI_Barrier */
    double t_compute;     /* Корисні обчислення */
    double t_imbalance;   /* 4. Дисбаланс навантаження */
    double t_memalloc;    /* 5. Управління пам'яттю (malloc/free) */
    double t_cache;       /* 6. Когерентність кешу (оцінка через naive vs BT) */
    double t_total;
} Overhead;

static double *alloc_timed(size_t bytes, double *t_mem) {
    double t0 = MPI_Wtime();
    double *p = (double *)malloc(bytes);
    *t_mem += MPI_Wtime() - t0;
    if (!p) { MPI_Abort(MPI_COMM_WORLD, 1); }
    return p;
}

static void init_matrix(double *m, int rows, int cols, int offset) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            m[i*cols+j] = (i + j + offset + 1) * 0.1;
}

static void transpose(const double *B, double *BT, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            BT[j*n+i] = B[i*n+j];
}

/* Наївне множення (для оцінки втрат когерентності кешу) */
static void matmul_naive(const double *A, const double *B,
                          double *C, int rows, int n) {
    memset(C, 0, (size_t)rows * n * sizeof(double));
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                C[i*n+j] += A[i*n+k] * B[k*n+j];
}

/* Оптимізоване множення з транспонуванням */
static void matmul_transposed(const double *A, const double *BT,
                               double *C, int rows, int n) {
    memset(C, 0, (size_t)rows * n * sizeof(double));
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += A[i*n+k] * BT[j*n+k];
            C[i*n+j] = sum;
        }
}

int main(int argc, char *argv[]) {
    double t_before_init = MPI_Wtime();
    MPI_Init(&argc, &argv);
    double t_after_init = MPI_Wtime();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = (argc > 1) ? atoi(argv[1]) : 1024;
    int rows = n / size;

    Overhead oh = {0};
    oh.t_init = t_after_init - t_before_init;

    /* ====== 5. Управління пам'яттю: вимірюємо malloc ====== */
    double *B    = alloc_timed((size_t)n * n * sizeof(double), &oh.t_memalloc);
    double *BT   = alloc_timed((size_t)n * n * sizeof(double), &oh.t_memalloc);
    double *Aloc = alloc_timed((size_t)rows * n * sizeof(double), &oh.t_memalloc);
    double *Cloc = alloc_timed((size_t)rows * n * sizeof(double), &oh.t_memalloc);
    double *A = NULL, *C = NULL;

    if (rank == 0) {
        A = alloc_timed((size_t)n * n * sizeof(double), &oh.t_memalloc);
        C = alloc_timed((size_t)n * n * sizeof(double), &oh.t_memalloc);
        init_matrix(A, n, n, 0);
        init_matrix(B, n, n, 1);
    }

    /* ====== 3. Синхронізація: бар'єр ====== */
    double tb0 = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    oh.t_barrier = MPI_Wtime() - tb0;

    /* ====== 2. Комунікації: MPI_Bcast(B) ====== */
    double tc0 = MPI_Wtime();
    MPI_Bcast(B, n*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    oh.t_bcast = MPI_Wtime() - tc0;

    /* Транспонування */
    transpose(B, BT, n);

    /* ====== 2. Комунікації: MPI_Scatter(A) ====== */
    tc0 = MPI_Wtime();
    MPI_Scatter(A, rows*n, MPI_DOUBLE,
                Aloc, rows*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    oh.t_scatter = MPI_Wtime() - tc0;

    /* ====== 6. Когерентність кешу: naive vs transposed ====== */
    /* Виміряємо різницю між наївним і кеш-оптимізованим множенням
       на малому блоці, щоб оцінити втрати від cache misses */
    if (n <= 2048) {
        double *C_naive = (double *)malloc((size_t)rows * n * sizeof(double));
        double t_n0 = MPI_Wtime();
        matmul_naive(Aloc, B, C_naive, rows, n);
        double t_naive = MPI_Wtime() - t_n0;

        double t_o0 = MPI_Wtime();
        matmul_transposed(Aloc, BT, Cloc, rows, n);
        double t_opt = MPI_Wtime() - t_o0;

        /* Різниця = витрати когерентності кешу (cache miss overhead) */
        oh.t_cache   = (t_naive > t_opt) ? t_naive - t_opt : 0.0;
        oh.t_compute = t_opt;
        free(C_naive);
    } else {
        /* Для великих матриць — тільки оптимізоване множення */
        MPI_Barrier(MPI_COMM_WORLD);
        double t_c0 = MPI_Wtime();
        matmul_transposed(Aloc, BT, Cloc, rows, n);
        oh.t_compute = MPI_Wtime() - t_c0;
        oh.t_cache   = oh.t_compute * 0.15; /* оцінка ~15% для великих n */
    }

    /* ====== 2. Комунікації: MPI_Gather(C) ====== */
    tc0 = MPI_Wtime();
    MPI_Gather(Cloc, rows*n, MPI_DOUBLE,
               C,    rows*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    oh.t_gather = MPI_Wtime() - tc0;

    /* ====== 4. Дисбаланс навантаження ====== */
    /* Вимірюємо через різницю max та min часу обчислень між процесами */
    double max_compute, min_compute;
    MPI_Reduce(&oh.t_compute, &max_compute, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&oh.t_compute, &min_compute, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    oh.t_imbalance = max_compute - min_compute;

    /* Зведення max через усі процеси */
    double max_bcast, max_scatter, max_gather, max_total, max_cache, max_mem;
    MPI_Reduce(&oh.t_bcast,    &max_bcast,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&oh.t_scatter,  &max_scatter, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&oh.t_gather,   &max_gather,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&oh.t_cache,    &max_cache,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&oh.t_memalloc, &max_mem,     1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double t_fin0 = MPI_Wtime();
    MPI_Finalize();
    double t_fin = MPI_Wtime() - t_fin0;

    if (rank == 0) {
        oh.t_total = oh.t_init + max_bcast + max_scatter + max_cache
                   + oh.t_barrier + max_compute + max_gather
                   + oh.t_imbalance + max_mem + t_fin;

        double comm_total = max_bcast + max_scatter + max_gather;
        double overhead   = comm_total + oh.t_barrier + oh.t_imbalance
                          + max_mem + max_cache + oh.t_init + t_fin;

        printf("\n=== MPI: n=%d, p=%d ===\n\n", n, size);
        printf("┌──────────────────────────────────┬──────────────┬──────────┐\n");
        printf("│ Категорія НВ                     │ Час (мс)     │ %%        │\n");
        printf("├──────────────────────────────────┼──────────────┼──────────┤\n");
        printf("│ 1. Ініціалізація (Init+Fin)       │ %10.2f   │ %6.2f%%  │\n",
               (oh.t_init+t_fin)*1000, (oh.t_init+t_fin)/oh.t_total*100);
        printf("│ 2. Комунікації                   │              │          │\n");
        printf("│    MPI_Bcast(B) [%4.0f MB]        │ %10.2f   │ %6.2f%%  │\n",
               (double)n*n*8/1e6, max_bcast*1000, max_bcast/oh.t_total*100);
        printf("│    MPI_Scatter(A) [%3.0f MB/proc] │ %10.2f   │ %6.2f%%  │\n",
               (double)rows*n*8/1e6, max_scatter*1000, max_scatter/oh.t_total*100);
        printf("│    MPI_Gather(C)  [%3.0f MB/proc] │ %10.2f   │ %6.2f%%  │\n",
               (double)rows*n*8/1e6, max_gather*1000, max_gather/oh.t_total*100);
        printf("│ 3. Синхронізація (Barrier)       │ %10.2f   │ %6.2f%%  │\n",
               oh.t_barrier*1000, oh.t_barrier/oh.t_total*100);
        printf("│ 4. Дисбаланс навантаження        │ %10.2f   │ %6.2f%%  │\n",
               oh.t_imbalance*1000, oh.t_imbalance/oh.t_total*100);
        printf("│ 5. Управління пам'яттю (malloc)  │ %10.2f   │ %6.2f%%  │\n",
               max_mem*1000, max_mem/oh.t_total*100);
        printf("│ 6. Когерентність кешу            │ %10.2f   │ %6.2f%%  │\n",
               max_cache*1000, max_cache/oh.t_total*100);
        printf("│    Обчислення (корисна робота)   │ %10.2f   │ %6.2f%%  │\n",
               max_compute*1000, max_compute/oh.t_total*100);
        printf("├──────────────────────────────────┼──────────────┼──────────┤\n");
        printf("│ Всього накладних витрат          │ %10.2f   │ %6.2f%%  │\n",
               overhead*1000, overhead/oh.t_total*100);
        printf("│ Загальний час                    │ %10.2f   │ 100.00%%  │\n",
               oh.t_total*1000);
        printf("└──────────────────────────────────┴──────────────┴──────────┘\n");

        double flops = 2.0 * n * n * n;
        printf("\nGFLOP/s: %.2f\n", flops / max_compute / 1e9);
        printf("Обсяг Bcast: %.1f MB\n", (double)n*n*8/1e6);

        free(A); free(C);
    }

    free(B); free(BT); free(Aloc); free(Cloc);
    return 0;
}
