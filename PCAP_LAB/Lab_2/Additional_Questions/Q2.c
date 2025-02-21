#include <mpi.h>
#include <stdio.h>

int factorial(int x){
    if(x <= 1) return 1;
    else return x * factorial(x-1);
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int temp = rank+1;
    if(!(rank % 2)){
        // Do (rank*2 + 1)!
        printf("Rank = %d Ans = " , rank);
        printf("%d\n" , factorial(temp));
    }
    else{
        // Do n*(n+1)/2
        printf("Rank = %d Ans = " , rank);
        printf("%d\n" , (temp * (temp + 1)) / 2);
    }
    MPI_Finalize();
    return 0;
}
