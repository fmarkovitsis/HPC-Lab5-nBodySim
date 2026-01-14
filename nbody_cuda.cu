#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "custom_sem.h"
#include "timer.h"
#include "gputimer.h"

#define SOFTENING 0.01f
#define ACCURACY 0.001f

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_RESET   "\x1b[0m"

#define BLOCK_SIZE 1024

typedef struct {
    float x, y, z, vx, vy, vz;
} Body;

int bodies_per_system = 8192;       /* Number of bodies per galaxy */
const float dt = 0.01f;
int nIters = 20;                    /* Simulation steps */

int ndev = -1;
dim3 grid_dim;
int *thread_args;
int bytes, total_bodies;
int bytes_per_field;

int ctx_counter = 0;

Body *data; // Host AoS buffer (for file I/O)

float *h_x, *h_y, *h_z, *h_vx, *h_vy, *h_vz;

my_bin_semT *gpu_semaphores;
my_bin_semT main_sem;
my_bin_semT mtx;


__global__ void bodyForce (float* x_arr, float* y_arr, float* z_arr, 
                           float* vx_arr, float* vy_arr, float* vz_arr,
                           float* temp_x_arr, float* temp_y_arr, float* temp_z_arr,
                           int n, float dt)
{
    int i, j;
    int tiles;
    int x_idx, y_idx, bodyIdx;
    float Fx, Fy, Fz, dx, dy, dz, distSqr, invDist, invDist3;
    
    // Shared memory now stores simple floats, not structs
    __shared__ float cache_x[BLOCK_SIZE];
    __shared__ float cache_y[BLOCK_SIZE];
    __shared__ float cache_z[BLOCK_SIZE];

    tiles = gridDim.x;
    x_idx = threadIdx.x;
    y_idx = blockIdx.y; // System index

    // Calculate global index for this thread
    bodyIdx = blockIdx.y * gridDim.x * BLOCK_SIZE + blockIdx.x * BLOCK_SIZE + x_idx;

    float my_x = x_arr[bodyIdx];
    float my_y = y_arr[bodyIdx];
    float my_z = z_arr[bodyIdx];
    float my_vx = vx_arr[bodyIdx];
    float my_vy = vy_arr[bodyIdx];
    float my_vz = vz_arr[bodyIdx];

    Fx = 0.0f;
    Fy = 0.0f;
    Fz = 0.0f;

    // Tile loop
    for (i = 0; i < tiles; i++)
    {
        int tile_offset = y_idx * n + i * BLOCK_SIZE + x_idx;
        cache_x[x_idx] = x_arr[tile_offset];
        cache_y[x_idx] = y_arr[tile_offset];
        cache_z[x_idx] = z_arr[tile_offset];

        __syncthreads();

        // Computation
        #pragma unroll 32
        for (j = 0; j < BLOCK_SIZE; j++) {
            dx = cache_x[j] - my_x;
            dy = cache_y[j] - my_y;
            dz = cache_z[j] - my_z;
            distSqr = fmaf(dx, dx, fmaf(dy, dy, fmaf(dz, dz, SOFTENING)));
            invDist = rsqrtf(distSqr); 
            invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        __syncthreads();
    }

    // Update velocities
    my_vx += dt * Fx;
    my_vy += dt * Fy;
    my_vz += dt * Fz;

    // Update positions
    my_x += my_vx * dt;
    my_y += my_vy * dt;
    my_z += my_vz * dt;

    temp_x_arr[bodyIdx]  = my_x;
    temp_y_arr[bodyIdx]  = my_y;
    temp_z_arr[bodyIdx]  = my_z;
    vx_arr[bodyIdx] = my_vx;
    vy_arr[bodyIdx] = my_vy;
    vz_arr[bodyIdx] = my_vz;
}

void *createContextGPU(void *args)
{
    // SoA Pointers for Device
    float *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz;
    float *temp_d_x, *temp_d_y, *temp_d_z;
    float *temp;
    int iter;
    cudaError_t err;

    int i = *(int*) args;

    cudaSetDevice(i);
    cudaFree(0); 

    mysem_down(&mtx);
    ctx_counter++;
    if (ctx_counter == ndev)
    {
        mysem_up(&main_sem);
    }
    mysem_up(&mtx);

    mysem_down(&gpu_semaphores[i]);

    int bodies_per_gpu = total_bodies / ndev;
    int bytes_slice = bodies_per_gpu * sizeof(float);
    int offset = i * bodies_per_gpu; // Offset in the global arrays

    if (cudaMalloc((void **) &d_x, bytes_slice) != cudaSuccess) printf("Malloc x failed %d\n", i);
    if (cudaMalloc((void **) &d_y, bytes_slice) != cudaSuccess) printf("Malloc y failed %d\n", i);
    if (cudaMalloc((void **) &d_z, bytes_slice) != cudaSuccess) printf("Malloc z failed %d\n", i);
    if (cudaMalloc((void **) &d_vx, bytes_slice) != cudaSuccess) printf("Malloc vx failed %d\n", i);
    if (cudaMalloc((void **) &d_vy, bytes_slice) != cudaSuccess) printf("Malloc vy failed %d\n", i);
    if (cudaMalloc((void **) &d_vz, bytes_slice) != cudaSuccess) printf("Malloc vz failed %d\n", i);
    if (cudaMalloc((void **) &temp_d_x, bytes_slice) != cudaSuccess) printf("Malloc temp_d_x failed %d\n", i);
    if (cudaMalloc((void **) &temp_d_y, bytes_slice) != cudaSuccess) printf("Malloc temp_d_y failed %d\n", i);
    if (cudaMalloc((void **) &temp_d_z, bytes_slice) != cudaSuccess) printf("Malloc temp_d_z failed %d\n", i);

    cudaMemcpy(d_x, &h_x[offset], bytes_slice, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, &h_y[offset], bytes_slice, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, &h_z[offset], bytes_slice, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, &h_vx[offset], bytes_slice, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, &h_vy[offset], bytes_slice, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, &h_vz[offset], bytes_slice, cudaMemcpyHostToDevice);

    for (iter = 1; iter <= nIters; iter++) {
        bodyForce<<<grid_dim, BLOCK_SIZE>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, temp_d_x, temp_d_y, temp_d_z, bodies_per_system, dt);
        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s in %s, line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        }

        temp = d_x; d_x = temp_d_x; temp_d_x = temp;
        temp = d_y; d_y = temp_d_y; temp_d_y = temp;
        temp = d_z; d_z = temp_d_z; temp_d_z = temp;
    }

    cudaMemcpy(&h_x[offset], d_x, bytes_slice, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_y[offset], d_y, bytes_slice, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_z[offset], d_z, bytes_slice, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_vx[offset], d_vx, bytes_slice, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_vy[offset], d_vy, bytes_slice, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_vz[offset], d_vz, bytes_slice, cudaMemcpyDeviceToHost);

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
    cudaFree(temp_d_x); cudaFree(temp_d_y); cudaFree(temp_d_z);

    return NULL;
}


int main(const int argc, const char *argv[]) {
    /* Default Configuration */
    int num_systems = 32;               

    FILE *fp;
    float *buf;
    double totalTime;
    int i;

    pthread_t *threads;

    cudaError_t err = cudaGetDeviceCount(&ndev);
    printf("cudaGetDeviceCount err=%s, ndev=%d\n",cudaGetErrorString(err), ndev);

    threads = (pthread_t*) malloc (ndev * sizeof(pthread_t));
    thread_args = (int *) malloc (ndev * sizeof(int));
    for (i = 0; i < ndev; i++)
    {
        thread_args[i] = i;
    }

    gpu_semaphores = (my_bin_semT *) malloc (ndev*sizeof(my_bin_semT));
    for (i=0; i<ndev; i++)
    {
        mysem_init(&gpu_semaphores[i], 0);
    }
    mysem_init(&mtx, 1);
    mysem_init(&main_sem, 0);
    
    for (i = 0; i < ndev; i++)
    {
        if (pthread_create(&threads[i], NULL, &createContextGPU, (void*)&thread_args[i]) != 0)
            return(1);
    }

    /* Attempt to load dataset */
    fp = fopen("galaxy_data.bin", "rb");
    if (fp) {
            fread(&num_systems, sizeof(int), 1, fp);
            fread(&bodies_per_system, sizeof(int), 1, fp);
            printf("Found dataset: %d systems of %d bodies.\n", num_systems,
                    bodies_per_system);
    } else {
            printf("No dataset found. Using random initialization.\n");
    }

    /* Allocate memory for ALL systems (Host AoS for I/O) */
    total_bodies = num_systems * bodies_per_system;
    bytes = total_bodies * sizeof(Body);
    data = (Body *) malloc(bytes);

    /* Initialize data */
    if (fp) {
            fread(data, sizeof(Body), total_bodies, fp);
            fclose(fp);
    } else {
            buf = (float *) data;
            for (int i = 0; i < 6 * total_bodies; i++) {
                buf[i] = 2.0f * (rand() / (float) RAND_MAX) - 1.0f;
        }
    }

    bytes_per_field = total_bodies * sizeof(float);
    h_x = (float*) malloc(bytes_per_field);
    h_y = (float*) malloc(bytes_per_field);
    h_z = (float*) malloc(bytes_per_field);
    h_vx = (float*) malloc(bytes_per_field);
    h_vy = (float*) malloc(bytes_per_field);
    h_vz = (float*) malloc(bytes_per_field);

    // "Transpose" AoS to SoA
    for (int i = 0; i < total_bodies; i++) {
        h_x[i] = data[i].x;
        h_y[i] = data[i].y;
        h_z[i] = data[i].z;
        h_vx[i] = data[i].vx;
        h_vy[i] = data[i].vy;
        h_vz[i] = data[i].vz;
    }

    grid_dim.x = bodies_per_system/BLOCK_SIZE;
    grid_dim.y = num_systems/ndev;

    mysem_down(&main_sem);
    
    totalTime = 0.0;
    StartTimer();
    
    for (i = 0; i < ndev; i++)
    {
        mysem_up(&gpu_semaphores[i]);
    }
    
    for (i = 0; i < ndev; i++)
    {
        pthread_join(threads[i], NULL);
    }

    totalTime = GetTimer() / 1000.0;
    printf("Total Comp Time: %.3f seconds\n", totalTime);
    double interactions_per_system = (double) bodies_per_system * bodies_per_system;
    double total_interactions = interactions_per_system * num_systems * nIters;
    printf("Average Throughput: %0.3f Billion Interactions / second\n",
           1e-9 * total_interactions / totalTime);

    #pragma omp parallel for
    for (i = 0; i < ndev; i++)
    {
        cudaSetDevice(i);
        cudaFree(0);
        cudaDeviceReset();
    }

    for (int i = 0; i < total_bodies; i++) {
        data[i].x = h_x[i];
        data[i].y = h_y[i];
        data[i].z = h_z[i];
        data[i].vx = h_vx[i];
        data[i].vy = h_vy[i];
        data[i].vz = h_vz[i];
    }

    FILE *f_out = fopen("final_GPU.bin", "wb");
    fwrite(&num_systems, sizeof(int), 1, f_out);
    fwrite(&bodies_per_system, sizeof(int), 1, f_out);
    fwrite(data, sizeof(Body), total_bodies,f_out);
    fclose(f_out);
    printf(ANSI_COLOR_GREEN "Dumped successfully.\n" ANSI_COLOR_RESET);

    free(data);
    free(h_x); free(h_y); free(h_z);
    free(h_vx); free(h_vy); free(h_vz);

    for (i = 0; i < ndev; i++)
    {
        mysem_destroy(&gpu_semaphores[i]);
    }
    mysem_destroy(&main_sem);
    mysem_destroy(&mtx);

    return 0;
}