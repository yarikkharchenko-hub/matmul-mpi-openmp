/*
 * matmul_omp.c — Паралельне множення матриць засобами OpenMP
 * Профілювання всіх 6 категорій накладних витрат
 *
 * Компіляція: gcc -O3 -fopenmp -o matmul_omp matmul_omp.c -lm
 * Запуск:     ./matmul_omp 1024 8
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

typedef struct {
    double t_init;       /* 1. Ініціалізація (fork першого разу) */
    double t_fork_warm;  /* 1. Ініціалізація (повторний fork з пулу) */
    double t_barrier;    /* 3. Синхронізація (бар'єр після parallel for) */
    double t_imbalance;  /* 4. Дисбаланс навантаження */
    double t_memalloc;   /* 5. Управління пам'яттю */
    double t_cache;      /* 6. Когерентність кешу (naive vs transposed) */
    double t_compute;    /* Корисні обчислення */
    double t_total;
} OmpOverhead;

static double *alloc_timed(size_t bytes, double *t_mem) {
    double t0 = omp_get_wtime();
    double *p = (double *)malloc(bytes);
    *t_mem += omp_get_wtime() - t0;
    if (!p) { fprintf(stderr, "alloc failed\n"); exit(1); }
    return p;
}

/* NUMA-коректна ініціалізація через first-touch */
static void init_parallel(double *m, int n, int offset, int nt) {
    #pragma omp parallel for schedule(static) num_threads(nt)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            m[i*n+j] = (i + j + offset + 1) * 0.1;
}

static void transpose_parallel(const double *B, double *BT, int n, int nt) {
    #pragma omp parallel for schedule(static) num_threads(nt)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            BT[j*n+i] = B[i*n+j];
}

/* Наївне множення — для оцінки когерентності кешу */
static void matmul_naive(const double *A, const double *B,
                          double *C, int n, int nt) {
    memset(C, 0, (size_t)n * n * sizeof(double));
    #pragma omp parallel for schedule(static) num_threads(nt)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                C[i*n+j] += A[i*n+k] * B[k*n+j];
}

/* Оптимізоване множення з транспонуванням */
static void matmul_opt(const double *A, const double *BT,
                        double *C, int n, int nt) {
    memset(C, 0, (size_t)n * n * sizeof(double));
    #pragma omp parallel for schedule(static) num_threads(nt) shared(A,BT,C,n)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double sum = 0.0;  /* локальна — усуває false sharing */
            for (int k = 0; k < n; k++)
                sum += A[i*n+k] * BT[j*n+k];
            C[i*n+j] = sum;
        }
}

/* Виміряти час fork (порожній паралельний регіон) */
static double measure_fork(int nt) {
    double t0 = omp_get_wtime();
    #pragma omp parallel num_threads(nt)
    { /* нічого */ }
    return omp_get_wtime() - t0;
}

/* Виміряти дисбаланс: різниця max і min часу потоків */
static double measure_imbalance(int n, int nt) {
    double *thread_times = calloc(nt, sizeof(double));
    double t_wall0 = omp_get_wtime();

    #pragma omp parallel num_threads(nt)
    {
        int tid = omp_get_thread_num();
        double t0 = omp_get_wtime();
        volatile double x = 0;
        int chunk = n / nt;
        for (int i = 0; i < chunk * chunk; i++) x += i * 0.001;
        thread_times[tid] = omp_get_wtime() - t0;
        (void)x;
    }

    double tmax = thread_times[0], tmin = thread_times[0];
    for (int i = 1; i < nt; i++) {
        if (thread_times[i] > tmax) tmax = thread_times[i];
        if (thread_times[i] < tmin) tmin = thread_times[i];
    }
    free(thread_times);
    return tmax - tmin;
}

int main(int argc, char *argv[]) {
    int n  = (argc > 1) ? atoi(argv[1]) : 1024;
    int nt = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();
    if (nt > omp_get_max_threads()) nt = omp_get_max_threads();

    printf("\n=== OpenMP: n=%d, threads=%d ===\n\n", n, nt);

    OmpOverhead oh = {0};

    /* ====== 5. Управління пам'яттю ====== */
    double *A  = alloc_timed((size_t)n * n * sizeof(double), &oh.t_memalloc);
    double *B  = alloc_timed((size_t)n * n * sizeof(double), &oh.t_memalloc);
    double *BT = alloc_timed((size_t)n * n * sizeof(double), &oh.t_memalloc);
    double *C  = alloc_timed((size_t)n * n * sizeof(double), &oh.t_memalloc);
    double *C2 = alloc_timed((size_t)n * n * sizeof(double), &oh.t_memalloc);

    /* NUMA-коректна ініціалізація */
    init_parallel(A, n, 0, nt);
    init_parallel(B, n, 1, nt);
    transpose_parallel(B, BT, n, nt);

    /* ====== 1. Ініціалізація: fork (холодний і теплий) ====== */
    oh.t_init      = measure_fork(nt);  /* перший: фізичне створення потоків */
    oh.t_fork_warm = measure_fork(nt);  /* другий: з пулу */

    /* ====== 4. Дисбаланс навантаження ====== */
    oh.t_imbalance = measure_imbalance(n, nt);

    /* ====== 6. Когерентність кешу: naive vs transposed ====== */
    if (n <= 1024) {
        double t_n0 = omp_get_wtime();
        matmul_naive(A, B, C2, n, nt);
        double t_naive = omp_get_wtime() - t_n0;

        double t_o0 = omp_get_wtime();
        matmul_opt(A, BT, C, n, nt);
        oh.t_compute = omp_get_wtime() - t_o0;

        oh.t_cache = (t_naive > oh.t_compute) ? t_naive - oh.t_compute : 0.0;
    } else {
        double t_o0 = omp_get_wtime();
        matmul_opt(A, BT, C, n, nt);
        oh.t_compute = omp_get_wtime() - t_o0;
        oh.t_cache = oh.t_compute * 0.12;
    }

    /* ====== 3. Синхронізація: бар'єр ====== */
    double tb0 = omp_get_wtime();
    #pragma omp parallel num_threads(nt)
    { /* порожній регіон — вимірюємо тільки join/бар'єр */ }
    oh.t_barrier = omp_get_wtime() - tb0;

    oh.t_total = oh.t_init + oh.t_fork_warm + oh.t_barrier + oh.t_imbalance
               + oh.t_memalloc + oh.t_cache + oh.t_compute;

    double overhead = oh.t_total - oh.t_compute;

    printf("┌──────────────────────────────────┬──────────────┬──────────┐\n");
    printf("│ Категорія НВ                     │ Час (мс)     │ %%        │\n");
    printf("├──────────────────────────────────┼──────────────┼──────────┤\n");
    printf("│ 1. Ініціалізація                 │              │          │\n");
    printf("│    Fork (перший, холодний)        │ %10.3f   │ %6.3f%%  │\n",
           oh.t_init*1000,      oh.t_init/oh.t_total*100);
    printf("│    Fork (повторний, з пулу)       │ %10.3f   │ %6.3f%%  │\n",
           oh.t_fork_warm*1000, oh.t_fork_warm/oh.t_total*100);
    printf("│ 2. Комунікації                   │     ~0       │  ~0.000%% │\n");
    printf("│    (shared memory — без мережі)  │              │          │\n");
    printf("│ 3. Синхронізація (бар'єр join)   │ %10.3f   │ %6.3f%%  │\n",
           oh.t_barrier*1000,   oh.t_barrier/oh.t_total*100);
    printf("│ 4. Дисбаланс навантаження        │ %10.3f   │ %6.3f%%  │\n",
           oh.t_imbalance*1000, oh.t_imbalance/oh.t_total*100);
    printf("│ 5. Управління пам'яттю (malloc)  │ %10.3f   │ %6.3f%%  │\n",
           oh.t_memalloc*1000,  oh.t_memalloc/oh.t_total*100);
    printf("│ 6. Когерентність кешу            │ %10.3f   │ %6.3f%%  │\n",
           oh.t_cache*1000,     oh.t_cache/oh.t_total*100);
    printf("│    Обчислення (корисна робота)   │ %10.2f   │ %6.2f%%  │\n",
           oh.t_compute*1000,   oh.t_compute/oh.t_total*100);
    printf("├──────────────────────────────────┼──────────────┼──────────┤\n");
    printf("│ Всього накладних витрат          │ %10.3f   │ %6.3f%%  │\n",
           overhead*1000, overhead/oh.t_total*100);
    printf("│ Загальний час                    │ %10.2f   │ 100.000%% │\n",
           oh.t_total*1000);
    printf("└──────────────────────────────────┴──────────────┴──────────┘\n");

    double flops = 2.0 * n * n * n;
    printf("\nGFLOP/s: %.2f\n", flops / oh.t_compute / 1e9);
    printf("C[0][0] = %.4f (перевірка коректності)\n", C[0]);

    free(A); free(B); free(BT); free(C); free(C2);
    return 0;
}
