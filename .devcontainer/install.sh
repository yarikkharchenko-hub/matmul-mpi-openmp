#!/bin/bash
set -e

echo "📦 Встановлення залежностей..."
apt-get update -qq
apt-get install -y -qq \
    gcc \
    g++ \
    make \
    libopenmpi-dev \
    openmpi-bin \
    libomp-dev \
    htop \
    time \
    linux-tools-common \
    2>/dev/null || true

echo ""
echo "✅ Встановлено:"
gcc --version | head -1
mpirun --version | head -1
echo "OpenMP: $(gcc -fopenmp -dM -E - < /dev/null | grep _OPENMP | awk '{print $3}')"

echo ""
echo "🔨 Компіляція проекту..."
cd /workspaces/$(ls /workspaces/ | head -1)
make all

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🚀 Готово! Доступні команди:"
echo ""
echo "  make test          # Швидкий тест (n=512, всі версії)"
echo "  make benchmark     # Повний бенчмарк (n=1024)"
echo ""
echo "  ./matmul_seq 1024"
echo "  ./matmul_omp 1024 4"
echo "  mpirun -np 4 ./matmul_mpi 1024"
echo "  mpirun -np 2 ./matmul_hybrid 1024 2"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
