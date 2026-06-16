/*
 * matmul_seq.c — Послідовний алгоритм множення матриць
 * з оптимізацією кешування (транспонування B)
 *
 * Компіляція: gcc -O3 -o matmul_seq matmul_seq.c -lm
 * Запуск:     ./matmul_seq 1024
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

static double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Виділення матриці n×n у безперервному блоці пам'яті */
double *alloc_matrix(int n) {
    double *m = (double *)malloc((size_t)n * n * sizeof(double));
    if (!m) { fprintf(stderr, "alloc failed\n"); exit(1); }
    return m;
}

/* Ініціалізація: A[i][j] = (i+1)*0.1, B[i][j] = (j+1)*0.1 */
void init_matrix(double *m, int n, int offset) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            m[i*n+j] = (i + j + offset + 1) * 0.1;
}

/* Транспонування: BT[j][i] = B[i][j] */
void transpose(const double *B, double *BT, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            BT[j*n+i] = B[i*n+j];
}

/* Наївне множення O(n³) без оптимізації */
void matmul_naive(const double *A, const double *B, double *C, int n) {
    memset(C, 0, (size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                C[i*n+j] += A[i*n+k] * B[k*n+j];
}

/* Оптимізоване множення: C[i][j] = SUM(A[i][l] * BT[j][l]) */
void matmul_transposed(const double *A, const double *BT, double *C, int n) {
    memset(C, 0, (size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += A[i*n+k] * BT[j*n+k];  /* обидва рядки — cache friendly */
            C[i*n+j] = sum;
        }
}

/* Перевірка результату: max|C1-C2| < eps */
double check_diff(const double *C1, const double *C2, int n) {
    double maxdiff = 0.0;
    for (int i = 0; i < n*n; i++) {
        double d = fabs(C1[i] - C2[i]);
        if (d > maxdiff) maxdiff = d;
    }
    return maxdiff;
}

int main(int argc, char *argv[]) {
    int n = (argc > 1) ? atoi(argv[1]) : 512;
    if (n <= 0 || n > 8192) { fprintf(stderr, "n must be 1..8192\n"); return 1; }

    printf("=== Послідовне множення матриць n=%d ===\n", n);
    printf("Розмір даних: %.1f MB\n", 3.0 * n * n * sizeof(double) / 1e6);

    double *A  = alloc_matrix(n);
    double *B  = alloc_matrix(n);
    double *BT = alloc_matrix(n);
    double *C1 = alloc_matrix(n);
    double *C2 = alloc_matrix(n);

    init_matrix(A, n, 0);
    init_matrix(B, n, 1);

    /* Тест 1: наївний алгоритм */
    double t0 = get_time();
    matmul_naive(A, B, C1, n);
    double t_naive = get_time() - t0;
    printf("\nНаївний алгоритм:        %.3f с\n", t_naive);

    /* Тест 2: з транспонуванням */
    t0 = get_time();
    transpose(B, BT, n);
    double t_trans = get_time() - t0;

    t0 = get_time();
    matmul_transposed(A, BT, C2, n);
    double t_opt = get_time() - t0;
    printf("Транспонування B:         %.6f с\n", t_trans);
    printf("Оптимізований алгоритм:  %.3f с\n", t_opt);
    printf("Прискорення (кеш):       %.2fx\n", t_naive / t_opt);

    /* Перевірка коректності */
    double diff = check_diff(C1, C2, n);
    printf("\nМакс. похибка: %.2e %s\n", diff, diff < 1e-9 ? "(OK)" : "(ERROR!)");

    /* FLOP-продуктивність */
    double flops = 2.0 * n * n * n;
    printf("GFLOP/s:       %.2f\n", flops / t_opt / 1e9);

    free(A); free(B); free(BT); free(C1); free(C2);
    return 0;
}
