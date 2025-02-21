#include <mpi.h>
#include <stdio.h>

int isPrime(int x){
    if(x == 2) return 1;
    else if (x == 0 || x == 1 || x == -1) return 0;
    for(int i = 2; i < x; i++){
        if((x % i) == 0) return 0;
    }
    return 1;
}

void findPrimes(int start , int end){
    for(int i = start; i <= end;i++){
        if(isPrime(i)) printf("Prime Number : %d\n" , i);
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(!rank){
        findPrimes(0 , 50);
    }
    else{
        findPrimes(51 , 100);
    }
    fflush(stdout);
    MPI_Finalize();
    return 0;
}
