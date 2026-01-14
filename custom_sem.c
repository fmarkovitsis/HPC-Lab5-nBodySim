#include<stdio.h>
#include"custom_sem.h"
#include<semaphore.h>
#include<sys/types.h>
#include<sys/sem.h>
#include<sys/ipc.h>
#include<errno.h>
#include <sys/stat.h> 
#include <pthread.h>

static pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER; 

int mysem_init(my_bin_semT *sem, int n){
    int semid;

    if(sem->initialized == 1) {
        return(-1);
    } 
    
    if (n != 0 && n != 1) {
        return(0);
    }
    pthread_mutex_lock(&mtx);

    sem->initialized = 1;
    sem->value = n;
    sem->destoyed = 0;
    semid = semget(IPC_PRIVATE, 1, IPC_CREAT | 0666);
    if (semid == -1) {
        perror("semget error\n");
        return 44;
    }
    sem->semid = semid;

    semctl(semid, 0, SETVAL, sem->value);
    pthread_mutex_unlock(&mtx);

    return(1);

}

int mysem_down(my_bin_semT *sem) {
    struct sembuf sem_op;

    if(sem->initialized == 0) {
        return(-1);
    } 
    
    pthread_mutex_lock(&mtx);
   
    sem_op.sem_num = 0;  
    sem_op.sem_op = -1;  
    sem_op.sem_flg = 0;
    sem->value = 0;
    pthread_mutex_unlock(&mtx);
  
    if (semop(sem->semid, &sem_op, 1) == -1) {
        perror("semop failed");
        return 44;
    }
    return(1);

}


int mysem_up(my_bin_semT *sem) {
    struct sembuf sem_op;
    int value;

    if(sem->initialized == 0) {
        printf("Semaphore is not initialized.\n");
        return(-1);
    } 

    pthread_mutex_lock(&mtx);
    value  =  semctl(sem->semid, 0, GETVAL);
    sem->value = value;///////newwwww

    if(sem->value == 1) {
        printf("UP: Already 1!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");  
        pthread_mutex_unlock(&mtx);
        return(0);
    }
    
    sem_op.sem_num = 0;  
    sem_op.sem_op = 1;  
    sem_op.sem_flg = 0;  
    if (semop(sem->semid, &sem_op, 1) == -1) {
        perror("semop failed");
        return 44;
    }
    sem->value = 1;
    pthread_mutex_unlock(&mtx);

    return(1);

}


int mysem_destroy(my_bin_semT *sem) {

    if(sem->destoyed == 1) {
        return(-1);
    } 
    if(sem->initialized == 0) {
        return(-1);
    } 
    // if (semctl(sem->semid, 0, IPC_RMID) == -1) {
    //     perror("semctl IPC_RMID failed");
    //     return 44;
    // }    

    return(1);


}
