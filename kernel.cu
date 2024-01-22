
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

#define BOID_NUM 262144
#define FIELD_MAX 100

#define OUTPUT 100

#define SIM_NUM 20

#define AVOID_DIS 2
#define SENSE_DIS 5

#define BS 512

//1列分のセルの数
#define CELL_LENGTH 5
#define CELL_HIGH 20 //FIELD_MAX/CELL_LENGTH
#define CELL_WIDTH 20 //FIELD_MAX/CELL_LENGTH
#define CELL_NUM 400 //CELL_HIGH*CELL_WIDTH



//xor128のx
//rand関数で初期化
unsigned int x;

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


typedef struct {
    int start, end;
}Cell;

Boid boids[BOID_NUM];
__device__ Boid boidsDevice[BOID_NUM];
__device__ Cell cellDevice[CELL_NUM];
__device__ Boid bufferDevice[BOID_NUM];

unsigned int xor_shift128() {
    static unsigned int y = 362436069;
    static unsigned int z = 521288629;
    static unsigned int w = 88685123;
    unsigned int t;

    t = x ^ (x << 11);
    x = y; y = z; z = w;
    return w = (w ^ (w >> 19)) ^ (t ^ (t >> 8));
}

//calc cell id
__host__ __device__ int get_cellid(int boidID) {
    return (int)CELL_WIDTH * (int)(boidsDevice[boidID].pos.y / CELL_LENGTH) + (int)(boidsDevice[boidID].pos.x / CELL_LENGTH);
}

//get random number
float rnd_p() {
    return (xor_shift128() / (float)4294967296) * (FIELD_MAX - 1);
}

float rnd_v() {
    return (xor_shift128() / (float)4294967296) * 2.0 - 1;
}

__global__ void init_cll() {
    int ix = blockIdx.x * BS + threadIdx.x;

    cellDevice[ix].start = 0;
    cellDevice[ix].end = -1;
}

//cell link listの作成及び更新
__global__ void make_CLL() {
    int ix = blockIdx.x * BS + threadIdx.x;

    //cellのstartとendを求めてshared memoryに格納する

    if (ix == 0) {
        cellDevice[get_cellid(ix)].start = ix;
    }
    else if (ix == BOID_NUM - 1) {
        if (get_cellid(ix) != get_cellid(ix)) {
            cellDevice[get_cellid(ix)].end = ix - 1;
            cellDevice[get_cellid(ix)].start = ix;
        }
        cellDevice[get_cellid(ix)].end = ix;
    }
    else {
        if (get_cellid(ix) != get_cellid(ix)) {
            cellDevice[get_cellid(ix)].end = ix - 1;
            cellDevice[get_cellid(ix)].start = ix;
        }
    }
}

//エージェントの距離
__device__ float getDistance(int b1ID, int b2ID) {
    float dx = bufferDevice[b1ID].pos.x - bufferDevice[b2ID].pos.x;
    float dy = bufferDevice[b1ID].pos.y - bufferDevice[b2ID].pos.y;

    return (float)(sqrt(dx * dx + dy * dy));

}

//衝突の回避
__device__ void Separation(int index, int* neigh_table) {
    int i, j, k;
    int cellx = bufferDevice[index].pos.x / CELL_LENGTH;
    int celly = bufferDevice[index].pos.y / CELL_LENGTH;

    for (j = celly - 1; j <= celly + 1; j++) {
        if (j < 0 || CELL_HIGH <= j) continue;
        for (i = 0; i < 3; i++) {
            int neighx = cellx + neigh_table[i * 3 + cellx % 3];
            if (neighx < 0 || CELL_WIDTH <= neighx) continue;
            int neigh = CELL_WIDTH * j + neighx;
            for (k = cellDevice[neigh].start; k <= cellDevice[neigh].end; k++) {
                if (k < 0) break;
                if (getDistance(index, k) < AVOID_DIS) {
                    boidsDevice[index].vel.x -= bufferDevice[k].pos.x - bufferDevice[index].pos.x;
                    boidsDevice[index].vel.y -= bufferDevice[k].pos.y - bufferDevice[index].pos.y;
                }
            }
        }
    }

}

//速度調節
__device__ void Alignment(int index, int* neigh_table) {
    float ave_x = 0, ave_y = 0;

    int i, j, k;
    int cnt = 0;
    int cellx = bufferDevice[index].pos.x / CELL_LENGTH;
    int celly = bufferDevice[index].pos.y / CELL_LENGTH;

    for (j = celly - 1; j <= celly + 1; j++) {
        if (j < 0 || CELL_HIGH <= j) continue;
        for (i = 0; i < 3; i++) {
            int neighx = cellx + neigh_table[i * 3 + cellx % 3];
            if (neighx < 0 || CELL_WIDTH <= neighx) continue;
            int neigh = CELL_WIDTH * j + neighx;
            for (k = cellDevice[neigh].start; k <= cellDevice[neigh].end; k++) {
                if (k < 0) break;
                if ((k != index) &&(getDistance(index, k) < SENSE_DIS)) {
                    ave_x += bufferDevice[k].vel.x;
                    ave_y += bufferDevice[k].vel.y;
                    cnt++;
                }
            }
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
__device__ void Cohesion(int index, int* neigh_table) {
    float cx = 0, cy = 0;

    int i, j, k;
    int cnt = 0;
    int cellx = bufferDevice[index].pos.x / CELL_LENGTH;
    int celly = bufferDevice[index].pos.y / CELL_LENGTH;

    for (j = celly - 1; j <= celly + 1; j++) {
        if (j < 0 || CELL_HIGH <= j) continue;
        for (i = 0; i < 3; i++) {
            int neighx = cellx + neigh_table[i * 3 + cellx % 3];
            if (neighx < 0 || CELL_WIDTH <= neighx) continue;
            int neigh = CELL_WIDTH * j + neighx;
            for (k = cellDevice[neigh].start; k <= cellDevice[neigh].end; k++) {
                if (k < 0) break;
                if ((k != index) &&
                    (getDistance(index, k) < SENSE_DIS)) {
                    cx += bufferDevice[k].pos.x;
                    cy += bufferDevice[k].pos.y;
                    cnt++;
                }
            }
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

    //アクセステーブルの作成
    __shared__ int neigh_table[9];
    if (id < 3) {
        neigh_table[0] = -1;
        neigh_table[1] = 1;
        neigh_table[2] = 0;

        neigh_table[id + 3] = neigh_table[id] + 1;
        if (neigh_table[id + 3] > 1) {
            neigh_table[id + 3] -= 3;
        }
        id += 3;

        neigh_table[id + 3] = neigh_table[id] + 1;
        if (neigh_table[id + 3] > 1) {
            neigh_table[id + 3] -= 3;
        }
    }

    __syncthreads();

    Separation(ix, neigh_table); //Rule1
    Alignment(ix, neigh_table); //Rule2
    Cohesion(ix, neigh_table); //Rule3

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

__global__ void gen_sort_key_cellid(int* sort_key) {
    int ix = blockIdx.x * BS + threadIdx.x;
    sort_key[ix] = get_cellid(ix);

}

int main(int argc, char* argv[]) {

    int i, sim;
    int cell_num = CELL_WIDTH * CELL_HIGH;

    int seed = 0;
    srand(seed);

    x = rand();

    //Initialize
    for (i = 0; i < BOID_NUM; i++) {
        boids[i].pos.x = rnd_p();
        boids[i].pos.y = rnd_p();
        boids[i].vel.x = rnd_v();
        boids[i].vel.y = rnd_v();
        boids[i].id = i;
    }



    int* d_sort_keys_cellid;
    cudaMalloc(&d_sort_keys_cellid, sizeof(int) * BOID_NUM);

    void *boidsPtr,*cellPtr,*bufferPtr;
    cudaGetSymbolAddress(&boidsPtr, boidsDevice);
    cudaGetSymbolAddress(&cellPtr, cellDevice);
    cudaGetSymbolAddress(&bufferPtr, bufferDevice);
    cudaMemcpy(boidsPtr, boids, sizeof(Boid) * BOID_NUM, cudaMemcpyHostToDevice);

    time_t start, stop;
    start = clock();
    //simulation
    for (sim = 0; sim < SIM_NUM; sim++) {
        
        printf("Steps %d\n", sim);
        //Sort agent by cell id
        gen_sort_key_cellid <<< BOID_NUM / BS, BS >>> (d_sort_keys_cellid);
        thrust::device_ptr<Boid> dev_data_ptr((Boid*)boidsPtr);
        thrust::device_ptr<int> dev_keys_ptr(d_sort_keys_cellid);
        cudaThreadSynchronize();

        thrust::sort_by_key(dev_keys_ptr, dev_keys_ptr +  BOID_NUM, dev_data_ptr);

        //CLL
        init_cll <<< cell_num / BS + 1, BS >>> ();
        make_CLL <<< BOID_NUM / BS, BS >>> ();


        cudaMemcpy(bufferPtr, boidsPtr, sizeof(Boid) * BOID_NUM,cudaMemcpyDeviceToDevice);
        cudaThreadSynchronize();
        simulation << < BOID_NUM / BS, BS >> > ();

        /* for(i = 0; i < BOID_NUM; i++){ */
        /*   if(boids[i].id == OUTPUT){ */
        /* 	printf("%f %f\n", boids[i].x, boids[i].y); */
        /* 	break; */
        /*   } */
        /* } */

    }
    stop = clock();
    float milliseconds = 0;
    printf("%lf[s]\n", (double)(stop - start) / CLOCKS_PER_SEC);
    return 0;

}