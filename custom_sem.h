#ifdef __cplusplus
extern "C" {
#endif

struct my_bin_sem {
    int semid;
    int value;
    int initialized;
    int destoyed;
    
};
typedef struct my_bin_sem my_bin_semT;


int mysem_init(my_bin_semT *sem, int n);

int mysem_down(my_bin_semT *sem);

int mysem_up(my_bin_semT *sem);

int mysem_destroy(my_bin_semT *sem);

#ifdef __cplusplus
}
#endif