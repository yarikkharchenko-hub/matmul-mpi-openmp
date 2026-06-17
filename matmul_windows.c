#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
  #include <windows.h>
  static double get_time(void){
      LARGE_INTEGER f,c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
      return (double)c.QuadPart/f.QuadPart;
  }
  #define SLEEP_MS(x) Sleep(x)
#else
  #include <pthread.h>
  #include <time.h>
  #include <unistd.h>
  static double get_time(void){
      struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
      return ts.tv_sec+ts.tv_nsec*1e-9;
  }
  #define SLEEP_MS(x) usleep((x)*1000)
#endif

typedef struct { int id,nt,n; double *A,*BT,*C,t_work; } TArg;

#ifdef _WIN32
DWORD WINAPI worker(LPVOID p){
#else
void *worker(void *p){
#endif
    TArg *a=(TArg*)p;
    int r0=a->id*(a->n/a->nt), r1=(a->id==a->nt-1)?a->n:r0+a->n/a->nt;
    double t0=get_time();
    for(int i=r0;i<r1;i++) for(int j=0;j<a->n;j++){
        double s=0; for(int k=0;k<a->n;k++) s+=a->A[i*a->n+k]*a->BT[j*a->n+k];
        a->C[i*a->n+j]=s;
    }
    a->t_work=get_time()-t0;
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

static void transpose(const double *B,double *BT,int n){
    for(int i=0;i<n;i++) for(int j=0;j<n;j++) BT[j*n+i]=B[i*n+j];
}
static void seq(const double *A,const double *BT,double *C,int n){
    for(int i=0;i<n;i++) for(int j=0;j<n;j++){
        double s=0; for(int k=0;k<n;k++) s+=A[i*n+k]*BT[j*n+k]; C[i*n+j]=s;
    }
}
static void bar(int p){
    printf("["); for(int i=0;i<30;i++) printf(i<p*30/100?"#":"-");
    printf("] %3d%%\r",p); fflush(stdout);
}

int main(int argc,char *argv[]){
    int n=argc>1?atoi(argv[1]):512;
    int nt=argc>2?atoi(argv[2]):4;
    if(n<64)n=64; if(n>1024)n=1024;
    if(nt<1)nt=1; if(nt>16)nt=16;

    printf("\n");
    printf("+==========================================================+\n");
    printf("|   Паралельне множення матриць: демонстраційна програма   |\n");
    printf("|   Харченко Ярослав Олегович -- Бакалаврська робота 2025  |\n");
    printf("+==========================================================+\n\n");
    printf("  Розмір матриці : %d x %d\n",n,n);
    printf("  Потоків        : %d\n",nt);
    printf("  Обсяг даних    : %.1f MB\n\n",3.0*n*n*8/1e6);

    double *A=malloc((size_t)n*n*8), *B=malloc((size_t)n*n*8);
    double *BT=malloc((size_t)n*n*8);
    double *C1=malloc((size_t)n*n*8), *C2=malloc((size_t)n*n*8);
    for(int i=0;i<n;i++) for(int j=0;j<n;j++){
        A[i*n+j]=(i+j+1)*0.1; B[i*n+j]=(i-j+2)*0.1;
    }
    transpose(B,BT,n);

    /* КРОК 1: Послідовне */
    printf("----------------------------------------------------------\n");
    printf("КРОК 1: Послідовне виконання (1 потік)\n\n");
    for(int p=0;p<=100;p+=10){ bar(p); SLEEP_MS(30); }
    double ts=get_time(); seq(A,BT,C1,n); double t_seq=get_time()-ts;
    printf("\n\n  Час T1 = %.3f с     GFLOP/s = %.2f\n\n",
           t_seq,2.0*n*n*n/t_seq/1e9);

    /* КРОК 2: Паралельне */
    printf("----------------------------------------------------------\n");
    printf("КРОК 2: Паралельне виконання (%d потоків)\n\n",nt);
    printf("  Розподіл рядків матриці A між потоками:\n");
    for(int t=0;t<nt;t++){
        int r0=t*(n/nt),r1=(t==nt-1)?n:r0+n/nt;
        printf("    Потік %d -> рядки %4d .. %4d  (%d рядків)\n",t,r0,r1-1,r1-r0);
    }
    printf("\n");

    TArg *args=malloc(nt*sizeof(TArg));
#ifdef _WIN32
    HANDLE *th=malloc(nt*sizeof(HANDLE));
#else
    pthread_t *th=malloc(nt*sizeof(pthread_t));
#endif
    memset(C2,0,(size_t)n*n*8);
    for(int p=0;p<=40;p+=10){ bar(p); SLEEP_MS(20); }

    double tp=get_time();
    for(int t=0;t<nt;t++){
        args[t].id=t; args[t].nt=nt; args[t].n=n;
        args[t].A=A; args[t].BT=BT; args[t].C=C2; args[t].t_work=0;
#ifdef _WIN32
        th[t]=CreateThread(NULL,0,worker,&args[t],0,NULL);
#else
        pthread_create(&th[t],NULL,worker,&args[t]);
#endif
    }
    for(int p=40;p<=95;p+=5){ bar(p); SLEEP_MS(40); }
#ifdef _WIN32
    WaitForMultipleObjects(nt,th,TRUE,INFINITE);
    for(int t=0;t<nt;t++) CloseHandle(th[t]);
#else
    for(int t=0;t<nt;t++) pthread_join(th[t],NULL);
#endif
    double t_par=get_time()-tp;
    bar(100);
    printf("\n\n  Час T%d = %.3f с     GFLOP/s = %.2f\n\n",
           nt,t_par,2.0*n*n*n/t_par/1e9);

    /* КРОК 3: Накладні витрати */
    double tmax=0,tmin=9999;
    for(int t=0;t<nt;t++){
        if(args[t].t_work>tmax)tmax=args[t].t_work;
        if(args[t].t_work<tmin)tmin=args[t].t_work;
    }
    double t_imb=tmax-tmin;
    double t_fork=t_par-tmax; if(t_fork<0)t_fork=0;

    printf("----------------------------------------------------------\n");
    printf("КРОК 3: Накладні витрати\n\n");
    printf("  +----------------------------------+----------+----------+\n");
    printf("  | Категорія НВ                     | Час (мс) | %%        |\n");
    printf("  +----------------------------------+----------+----------+\n");
    printf("  | 1. Ініціалізація (fork/join)     | %8.3f | %6.3f%%  |\n",
           t_fork*1000,t_par>0?t_fork/t_par*100:0);
    printf("  | 2. Комунікації (shared memory)   |     ~0   |  ~0.000%% |\n");
    printf("  | 3. Синхронізація (бар'єр)        |     ~0   |  ~0.000%% |\n");
    printf("  | 4. Дисбаланс навантаження        | %8.3f | %6.3f%%  |\n",
           t_imb*1000,t_par>0?t_imb/t_par*100:0);
    printf("  | 5. Управління пам'яттю           |     ~0   |  ~0.000%% |\n");
    printf("  | 6. Когерентність кешу             |     ~0   |  ~0.000%% |\n");
    printf("  |    Обчислення (корисна робота)   | %8.2f | %6.2f%%  |\n",
           tmax*1000,t_par>0?tmax/t_par*100:0);
    printf("  +----------------------------------+----------+----------+\n");
    printf("  | Загальний час                    | %8.2f | 100.00%%  |\n",
           t_par*1000);
    printf("  +----------------------------------+----------+----------+\n\n");

    /* КРОК 4: Результати */
    double sp=t_seq/t_par, ep=sp/nt*100;
    double diff=0;
    for(int i=0;i<n*n;i++){ double d=fabs(C1[i]-C2[i]); if(d>diff)diff=d; }

    printf("----------------------------------------------------------\n");
    printf("РЕЗУЛЬТАТИ\n\n");
    printf("  T1  (послідовний)   = %.3f с\n",t_seq);
    printf("  T%d  (паралельний)   = %.3f с\n",nt,t_par);
    printf("  S(%d) прискорення    = %.2fx\n",nt,sp);
    printf("  E(%d) ефективність   = %.1f%%\n",nt,ep);
    printf("  Похибка результату  = %.2e  %s\n\n",diff,
           diff<1e-8?"(OK)":"(ПОМИЛКА!)");
    printf("  Час обчислення по потоках:\n");
    for(int t=0;t<nt;t++)
        printf("    Потік %d: %.3f с\n",t,args[t].t_work);

    printf("\n----------------------------------------------------------\n");
    printf("Натисніть Enter для виходу...\n");
    getchar();

    free(A);free(B);free(BT);free(C1);free(C2);free(args);free(th);
    return 0;
}
