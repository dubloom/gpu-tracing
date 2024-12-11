#include <iostream>
#include <cuda_runtime.h>
#include <tracing.h>
#include <cmath>

#define BLOCK_SIZE 16

__global__ void matrixMultiplyKernel(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void matrixMultiplyCPU(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

float calculateNorm(const float *matrix, int N) {
    float norm = 0.0f;
    for (int i = 0; i < N * N; ++i) {
        norm += matrix[i] * matrix[i];
    }
    return sqrt(norm);
}

int main() {
    create_and_set_active_span("matmul");
    const int N = 1024; // Taille de la matrice N x N
    const int matrixSize = N * N * sizeof(float);

    // Allocation de mémoire pour les matrices sur l'hôte (CPU)
    float *h_A = (float *)malloc(matrixSize);
    float *h_B = (float *)malloc(matrixSize);
    float *h_C = (float *)malloc(matrixSize);

    // Initialisation des matrices A et B
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocation de mémoire pour les matrices sur le périphérique (GPU)
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, matrixSize);
    cudaMalloc((void **)&d_B, matrixSize);
    cudaMalloc((void **)&d_C, matrixSize);

    // Copie des matrices A et B sur le GPU
    cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice);

    // Configuration de la grille et des blocs
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Lancement du kernel
    matrixMultiplyKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    // Attente de la fin de l'exécution du kernel
    cudaDeviceSynchronize();

    // Copie du résultat sur le CPU
    cudaMemcpy(h_C, d_C, matrixSize, cudaMemcpyDeviceToHost);

    // Calcul de la norme de la matrice résultante sur le GPU
    float gpuNorm = calculateNorm(h_C, N);

    // Vérification des résultats (facultatif, utilise le CPU comme référence)
    float *reference = (float *)malloc(matrixSize);
    matrixMultiplyCPU(h_A, h_B, reference, N);

    // Calcul de la norme de la matrice résultante sur le CPU
    float cpuNorm = calculateNorm(reference, N);

    // Affichage des normes
    std::cout << "Norme de la matrice (GPU) : " << gpuNorm << std::endl;
    std::cout << "Norme de la matrice (CPU) : " << cpuNorm << std::endl;

    // Libération de la mémoire
    free(h_A);
    free(h_B);
    free(h_C);
    free(reference);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    end_active_span();
    return 0;
}
