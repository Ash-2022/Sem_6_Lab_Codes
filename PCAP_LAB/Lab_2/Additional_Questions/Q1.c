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

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int recievedNum , flag;
    int inputArr[size];
    if(!rank){
        for(int i = 0; i < size;i++){
            printf("Enter Element %d : " , i);
            fflush(stdout);
            scanf("%d" , &inputArr[i]);
            MPI_Send(&inputArr[i] , 1 , MPI_INT , i , 0 , MPI_COMM_WORLD);
        }
    }
    MPI_Recv(&recievedNum , 1 , MPI_INT , 0 , 0 , MPI_COMM_WORLD , MPI_STATUS_IGNORE);
    flag = isPrime(recievedNum);
    MPI_Send(&flag , 1 , MPI_INT , 0 , 1 , MPI_COMM_WORLD);
    if(!rank){
        int flagBuf[size];
        for(int i = 0; i < size;i++){
            MPI_Recv(&flagBuf[i] , 1 , MPI_INT , i , 1 , MPI_COMM_WORLD , MPI_STATUS_IGNORE);
            if(flagBuf[i]) printf("%d is Prime\n" , inputArr[i]);
            fflush(stdout);
        }
    }
    MPI_Finalize();
    return 0;
}
