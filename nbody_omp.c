#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "timer.h"
#include <omp.h>

#define SOFTENING 0.01f
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_RESET   "\x1b[0m"
typedef struct {
    float x, y, z, vx, vy, vz;
} Body;

/* Update a single galaxy. Parameters:
    - array of bodies
    - time step
    - number of bodies
*/
void bodyForce(Body * p, float dt, int n) {
    int i, j;
    float Fx, Fy, Fz, dx, dy, dz, distSqr, invDist, invDist3;

    for (i = 0; i < n; i++) {
	    Fx = 0.0f;
    	Fy = 0.0f;
    	Fz = 0.0f;
        #pragma omp unroll partial(32)
    	for (j = 0; j < n; j++) {
	        dx = p[j].x - p[i].x;
            dy = p[j].y - p[i].y;
            dz = p[j].z - p[i].z;
            distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            invDist = 1.0f / sqrtf(distSqr);
            invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        p[i].vx += dt * Fx;
        p[i].vy += dt * Fy;
        p[i].vz += dt * Fz;
    }
}

/* Integrate positions.
    - array of bodies
    - time step
    - number of bodies
*/
void integrate(Body * p, float dt, int n) {
    int i;
    #pragma omp unroll partial(16)
    for (i = 0; i < n; i++) {
	    p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}

int main(const int argc, const char *argv[]) {
    /* Default Configuration */
    int num_systems = 32;       	/* Number of independent galaxies */
    int bodies_per_system = 8192;	/* Number of bodies per galaxy */
    int nIters = 20;            	/* Simulation steps */ 
    
    const float dt = 0.01f;
    FILE *fp;
    int total_bodies, bytes, sys, iter;
    Body *data, *system_ptr;
    float *buf;
    double totalTime, interactions_per_system, total_interactions;


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

    /* Allocate memory for ALL systems */
    total_bodies = num_systems * bodies_per_system;
    bytes = total_bodies * sizeof(Body);
    data = (Body *) malloc(bytes);

    /* Initialize data */
    if (fp) {
	    fread(data, sizeof(Body), total_bodies, fp);
	    fclose(fp);
    } else {
	/* Random initialization if file missing */
	    buf = (float *) data;
	    for (int i = 0; i < 6 * total_bodies; i++) {
	        buf[i] = 2.0f * (rand() / (float) RAND_MAX) - 1.0f;
        }
    }

    printf("Running sequential CPU simulation for %d systems...\n",
           num_systems);

    int threads_num;      
    if (num_systems < 56) //tailored for venus
        threads_num = num_systems;
    else
        threads_num = 56;


    totalTime = 0.0;

    StartTimer();

    /* Time-steps */
    for (iter = 1; iter <= nIters; iter++) {
	    /* Galaxies */
        //seperate loop cause 1 sim can have more than one systems
        #pragma omp parallel for num_threads(threads_num)
	    for (sys = 0; sys < num_systems; sys++) {
	        /* Calculate offset for the galaxy */
	        system_ptr = &data[sys * bodies_per_system];
	        
	        /* Compute forces & integrate for the galaxy */
	        bodyForce(system_ptr, dt, bodies_per_system); //Nested loop traversing the bodies (new - prev)
	        integrate(system_ptr, dt, bodies_per_system); //just one loop updating positions through bodies
        }
    }

    totalTime = GetTimer() / 1000.0;

    /* Metrics calculation */
    interactions_per_system = (double) bodies_per_system * bodies_per_system;
    total_interactions = interactions_per_system * num_systems * nIters;

    printf("Total Time: %.3f seconds\n", totalTime);
    printf("Average Throughput: %0.3f Billion Interactions / second\n",
           1e-9 * total_interactions / totalTime);

    /* Dump final state of System 0, Body 0 and 1 for verification comparison */
    // printf("Final position of System 0, Body 0: %.4f, %.4f, %.4f\n",
    //        data[0].x, data[0].y, data[0].z);
    // printf("Final position of System 0, Body 1: %.4f, %.4f, %.4f\n",
    //        data[1].x, data[1].y, data[1].z);

    FILE *f_out = fopen("final_CPU.bin", "wb");

    if (f_out == NULL) {
        printf(ANSI_COLOR_GREEN"Error bro elouses ti einai afta\n"ANSI_COLOR_RESET);
    } else {
        fwrite(&num_systems, sizeof(int), 1, f_out);
        fwrite(&bodies_per_system, sizeof(int), 1, f_out);
        fwrite(data, sizeof(Body), total_bodies,f_out);
        fclose(f_out);
        printf(ANSI_COLOR_GREEN"Dumped successfully\n"ANSI_COLOR_RESET);
    }


    free(data);
    return 0;
}