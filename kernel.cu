
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <device_functions.h>

#define BOID_NUM 1024
#define FIELD_MAX 100

#define OUTPUT 100

#define SIM_NUM 2000

#define AVOID_DIS 2
#define SENSE_DIS 5

#define BS 512

//1列分のセルの数
#define CELL_LENGTH 5
#define CELL_HIGH 20 //FIELD_MAX/CELL_LENGTH
#define CELL_WIDTH 20 //FIELD_MAX/CELL_LENGTH
#define CELL_NUM 400 //CELL_HIGH*CELL_WIDTH


typedef struct {
    float x, y;
}Vector2;

typedef struct {
    //point
    Vector2 pos;

    //velocity
    Vector2 vel;

    //id
    int id;

}Boid;

Boid boids[BOID_NUM];
__device__ Boid boidsDevice[BOID_NUM];
__device__ Boid bufferDevice[BOID_NUM];


//get random number
float rnd_p() {
    return (rand() / (float)RAND_MAX) * (FIELD_MAX - 1);
}

float rnd_v() {
    return (rand() / (float)RAND_MAX) * 2.0 - 1;
}


//エージェントの距離
__device__ float getDistance(Boid b1, Boid b2) {
    float dx = b1.pos.x - b2.pos.x;
    float dy = b1.pos.y - b2.pos.y;

    return (float)(sqrt(dx * dx + dy * dy));

}

//衝突の回避
__device__ void Separation(int index) {
    for (int k = 0; k <= BOID_NUM; k++) {
        if (getDistance(bufferDevice[index], bufferDevice[index]) < AVOID_DIS) {
            boidsDevice[index].vel.x -= bufferDevice[k].pos.x - bufferDevice[index].pos.x;
            boidsDevice[index].vel.y -= bufferDevice[k].pos.y - bufferDevice[index].pos.y;
        }
    }
}

//速度調節
__device__ void Alignment(int index) {
    float ave_x = 0, ave_y = 0;
    int cnt = 0;

    for (int k = 0; k <= BOID_NUM; k++) {
        if ((k != index) && (getDistance(bufferDevice[index], bufferDevice[index]) < SENSE_DIS)) {
            ave_x += bufferDevice[k].vel.x;
            ave_y += bufferDevice[k].vel.y;
            cnt++;
        }
    }

    if (cnt) {
        ave_x = ave_x / cnt;
        ave_y = ave_y / cnt;

        boidsDevice[index].vel.x += (ave_x - bufferDevice[index].vel.x) / 10;
        boidsDevice[index].vel.y += (ave_y - bufferDevice[index].vel.y) / 10;
    }
}

//群れの中心に移動
__device__ void Cohesion(int index) {
    float cx = 0, cy = 0;
    int cnt = 0;

    for (int k = 0; k <= BOID_NUM; k++) {
        if ((k != index) &&
            (getDistance(bufferDevice[index], bufferDevice[index]) < SENSE_DIS)) {
            cx += bufferDevice[k].pos.x;
            cy += bufferDevice[k].pos.y;
            cnt++;
        }
    }

    if (cnt) {
        cx = cx / cnt;
        cy = cy / cnt;

        boidsDevice[index].vel.x += (cx - bufferDevice[index].pos.x) / 100;
        boidsDevice[index].vel.y += (cy - bufferDevice[index].pos.y) / 100;
    }
}

__global__ void simulation() {

    int ix = blockIdx.x * BS + threadIdx.x;

    int id = threadIdx.x;


    Separation(ix); //Rule1
    Alignment(ix); //Rule2
    Cohesion(ix); //Rule3

    float v = (float)sqrt(boidsDevice[ix].vel.x * boidsDevice[ix].vel.x + boidsDevice[ix].vel.y * boidsDevice[ix].vel.y);

    if (v > SENSE_DIS) {
        boidsDevice[ix].vel.x = boidsDevice[ix].vel.x * SENSE_DIS / v;
        boidsDevice[ix].vel.y = boidsDevice[ix].vel.y * SENSE_DIS / v;
    }

    boidsDevice[ix].pos.x += boidsDevice[ix].vel.x;
    boidsDevice[ix].pos.y += boidsDevice[ix].vel.y;

    if (boidsDevice[ix].pos.x < 0 && boidsDevice[ix].vel.x < 0) {
        boidsDevice[ix].pos.x = 0;
        boidsDevice[ix].vel.x *= -1;
    }
    else if (boidsDevice[ix].pos.x > FIELD_MAX - 0.5 && boidsDevice[ix].vel.x > 0) {
        boidsDevice[ix].pos.x = FIELD_MAX - 0.500001;
        boidsDevice[ix].vel.x *= -1;
    }
    if (boidsDevice[ix].pos.y < 0 && boidsDevice[ix].vel.y < 0) {
        boidsDevice[ix].pos.y = 0;
        boidsDevice[ix].vel.y *= -1;
    }
    else if (boidsDevice[ix].pos.y > FIELD_MAX - 0.5 && boidsDevice[ix].vel.y > 0) {
        boidsDevice[ix].pos.y = FIELD_MAX - 0.500001;
        boidsDevice[ix].vel.y *= -1;
    }

}


int main(int argc, char* argv[]) {

    int i, sim;
    int cell_num = CELL_WIDTH * CELL_HIGH;
    srand(0);


    //Initialize
    for (i = 0; i < BOID_NUM; i++) {
        boids[i].pos.x = rnd_p();
        boids[i].pos.y = rnd_p();
        boids[i].vel.x = rnd_v();
        boids[i].vel.y = rnd_v();
    }



    int* d_sort_keys_cellid;
    cudaMalloc(&d_sort_keys_cellid, sizeof(int) * BOID_NUM);

    void *boidsPtr,*bufferPtr;
    cudaGetSymbolAddress(&boidsPtr, boidsDevice);
    cudaGetSymbolAddress(&bufferPtr, bufferDevice);
    cudaMemcpy(boidsPtr, boids, sizeof(Boid) * BOID_NUM, cudaMemcpyHostToDevice);

    time_t start, stop;
    start = clock();
    //simulation
    for (sim = 0; sim < SIM_NUM; sim++) {
        
        if(sim%100==0)printf("Steps %d\n", sim);

        //cudaMemcpy(bufferPtr, boidsPtr, sizeof(Boid) * BOID_NUM,cudaMemcpyDeviceToDevice);
        cudaThreadSynchronize();
        simulation <<< BOID_NUM / BS, BS >>> ();
        cudaThreadSynchronize();

    }
    stop = clock();
    float milliseconds = 0;
    printf("%lf[s]\n", (double)(stop - start) / CLOCKS_PER_SEC);
    return 0;

}